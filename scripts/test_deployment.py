"""Deployment contracts, isolated from the developer's data and model account."""
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import config, db, clock, rag, agent
from backend.main import app, _verify_password


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [patch.object(config, "DATABASE_URL", ""),
                        patch.object(config, "DB_PATH", Path(self.tmp.name) / "test.db"),
                        patch.object(config, "IS_VERCEL", False),
                        patch.object(config, "DEMO_MODE", False)]
        for p in self.patches:
            p.start()
        self.client = TestClient(app)
        self.client.__enter__()
        self.headers = self.register("alice")

    def register(self, username):
        r = self.client.post("/api/auth", json={"username": username, "password": "test-password"})
        self.assertEqual(r.status_code, 200, r.text)
        return {"Authorization": "Bearer " + r.json()["token"]}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_register_profile_login_and_new_process_connection(self):
        r = self.client.post("/api/onboarding", headers=self.headers,
                             json={"off_work_end": "18:00", "wake_time": "08:00", "cooking": "基本自己做"})
        self.assertEqual(r.status_code, 200)
        new_headers = self.register("alice")
        self.assertEqual(self.client.get("/api/profile", headers=new_headers).json()["profile"]["cooking"], "基本自己做")
        self.assertEqual(self.client.post("/api/login", json={"username":"alice","password":"wrong"}).status_code, 401)
        self.assertTrue(db.get_user_by_username("alice")["password_hash"].split("$",1)[1].startswith("pbkdf2_sha256:"))

    def test_next_day_history_and_user_isolation(self):
        uid = db.get_user_by_username("alice")["id"]
        self.client.post("/api/onboarding",headers=self.headers,json={"off_work_end":"18:00","wake_time":"08:00"})
        db.insert_daily_record(uid, clock.today(), "{}", json.dumps({"judgement":"今晚从简","stop":"23:00 休息"}), 2)
        tomorrow = clock.now() + timedelta(days=1)
        with patch.object(clock, "now", return_value=tomorrow):
            state=self.client.get("/api/state",headers=self.headers).json()
            self.assertIsNotNone(state["pending_feedback"])
            self.assertEqual(len(self.client.get("/api/history",headers=self.headers).json()["history"]),1)
        bob=self.register("bob")
        self.assertEqual(self.client.get("/api/history",headers=bob).json()["history"],[])

    def test_real_clock_and_disabled_controls(self):
        self.assertLess(abs((datetime.now()-clock.now()).total_seconds()), 12*3600+1)
        self.assertEqual(self.client.post("/api/clock",json={"advance_days":1}).status_code,403)
        self.assertFalse(self.client.get("/api/state",headers=self.headers).json()["demo_mode"])

    def test_model_failure_is_reported_and_not_saved_as_a_real_plan(self):
        self.client.post("/api/onboarding",headers=self.headers,json={"off_work_end":"18:00","wake_time":"08:00"})
        with patch.object(agent,"run_agent",return_value=(dict(agent.FALLBACK_ADVICE),{})):
            r=self.client.post("/api/offwork",headers=self.headers,json={})
        self.assertEqual(r.status_code,503,r.text)
        self.assertEqual(r.json()["code"],"AI_UNAVAILABLE")
        self.assertEqual(self.client.get("/api/history",headers=self.headers).json()["history"],[])

    def test_trace_persists_and_is_private(self):
        uid=db.get_user_by_username("alice")["id"]
        db.save_trace(uid,{"rounds":[{"output":"done"}],"duration_s":1,"final":{}})
        self.assertEqual(self.client.get("/api/agent_trace",headers=self.register("alice")).json()["trace"]["steps"],["给出结论"])
        self.assertIsNone(self.client.get("/api/agent_trace",headers=self.register("bob")).json()["trace"])

    def test_photo_persists_and_only_owner_can_delete(self):
        with patch("backend.llm.vision",return_value='{"名称":"米饭"}'):
            r=self.client.post("/api/meal/photo",headers=self.headers,files={"file":("meal.jpg",b"test-image","image/jpeg")})
        self.assertEqual(r.status_code,200,r.text)
        mid=r.json()["meal_id"]
        conn=db.get_conn()
        self.assertEqual(bytes(conn.execute("SELECT content FROM uploads").fetchone()["content"]),b"test-image")
        conn.close()
        self.assertEqual(self.client.post("/api/meal/delete",headers=self.register("bob"),json={"meal_id":mid}).status_code,400)
        self.assertEqual(self.client.post("/api/meal/delete",headers=self.headers,json={"meal_id":mid}).status_code,200)
        conn=db.get_conn()
        self.assertIsNone(conn.execute("SELECT content FROM uploads").fetchone())
        conn.close()

    def test_upload_limit_and_auth(self):
        self.assertEqual(self.client.get("/api/history").status_code,401)
        r=self.client.post("/api/meal/photo",headers=self.headers,files={"file":("big.jpg",b"a"*(2*1024*1024+1),"image/jpeg")})
        self.assertEqual(r.status_code,413)

    def test_vercel_requires_persistent_database(self):
        with patch.object(config,"IS_VERCEL",True):
            with self.assertRaises(RuntimeError):
                db.get_conn()

    def test_packaged_corpus_works_with_empty_user_database(self):
        rag._vectors=None
        self.assertTrue(rag._load())
        self.assertEqual(len(rag._chunks),len(rag._vectors))
        self.assertGreater(len(rag._chunks),100)


if __name__ == "__main__":
    unittest.main()
