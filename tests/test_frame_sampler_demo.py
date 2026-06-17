from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

import app.main as main_module


class FrameSamplerDemoPageTest(unittest.TestCase):
    def test_frame_sampler_demo_page_returns_200(self) -> None:
        client = TestClient(main_module.app)

        response = client.get("/frame-sampler-demo")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Frame Sampler Demo", response.text)
        self.assertIn("sampler-config-select", response.text)
        self.assertIn("/api/action-cleanliness/workflow-video-candidates", response.text)


if __name__ == "__main__":
    unittest.main()
