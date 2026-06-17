from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.main as main_module
from app.analysis import read_image
from app.roi_store import ConfigStore


class SetupVideoReferenceTest(unittest.TestCase):
    def test_setup_save_extracts_first_frame_from_uploaded_video(self) -> None:
        original_store = main_module.config_store

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "roi_configs").mkdir()
            (root / "reference_images").mkdir()
            test_store = ConfigStore(
                config_dir=root / "roi_configs",
                reference_dir=root / "reference_images",
                store_catalog_path=root / "stores.json",
            )
            main_module.config_store = test_store

            video_path = root / "reference.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                1.0,
                (32, 24),
            )
            writer.write(np.full((24, 32, 3), (0, 200, 0), dtype=np.uint8))
            writer.write(np.full((24, 32, 3), (200, 0, 0), dtype=np.uint8))
            writer.release()

            try:
                client = TestClient(main_module.app)
                with video_path.open("rb") as handle:
                    response = client.post(
                        "/setup/save",
                        data={
                            "store_name": "VideoStore",
                            "cctv_nickname": "FrontCam",
                            "rois_json": json.dumps(
                                [
                                    {
                                        "name": "T01",
                                        "x": 2,
                                        "y": 3,
                                        "width": 12,
                                        "height": 10,
                                    }
                                ]
                            ),
                        },
                        files={"reference_image": ("reference.avi", handle, "video/x-msvideo")},
                        follow_redirects=False,
                    )
            finally:
                main_module.config_store = original_store

            self.assertEqual(response.status_code, 303)
            config = test_store.load("VideoStore_FrontCam")
            self.assertEqual(config.reference_image_path, "reference_images/VideoStore_FrontCam.png")
            saved_reference = root / "reference_images" / "VideoStore_FrontCam.png"
            self.assertTrue(saved_reference.exists())
            saved_image = read_image(saved_reference)
            self.assertEqual(saved_image.shape[:2], (24, 32))
            self.assertGreater(float(saved_image[:, :, 1].mean()), 100.0)


if __name__ == "__main__":
    unittest.main()
