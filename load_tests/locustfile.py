import time
from locust import HttpUser, task, between

class VideoAnalyticsUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    def on_start(self):
        """Login at the start of each user session."""
        self.client.post("/api/auth/login/", {
            "username": "testuser",
            "password": "testpass"
        })

    @task(2)
    def test_people_count(self):
        """Test people counting endpoint."""
        with self.client.post(
            "/api/video-analytics/jobs/",
            json={
                "job_type": "people_count",
                "input_video": "test_video.mp4"
            },
            catch_response=True
        ) as response:
            if response.status_code == 201:
                time.sleep(0.5)  # Simulate checking job status
                self.client.get(f"/api/video-analytics/jobs/{response.json()['id']}/")

    @task(1)
    def test_anpr(self):
        """Test ANPR endpoint."""
        with self.client.post(
            "/api/video-analytics/jobs/",
            json={
                "job_type": "anpr",
                "input_video": "test_video.mp4"
            },
            catch_response=True
        ) as response:
            if response.status_code == 201:
                time.sleep(0.5)  # Simulate checking job status
                self.client.get(f"/api/video-analytics/jobs/{response.json()['id']}/")

    @task(1)
    def test_face_auth(self):
        """Test face authentication endpoint."""
        with self.client.post(
            "/api/face-auth/verify/",
            files={
                'image': open('test_face.jpg', 'rb')
            },
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure("Face auth failed")
