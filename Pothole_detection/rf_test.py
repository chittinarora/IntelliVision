from roboflow import Roboflow

print("👉 Starting Roboflow test...")

rf = Roboflow(api_key="vfMnQeFixryhPw18Thmz")
print("✅ API key accepted")

try:
    project = rf.workspace("brad-dwyer").project("pothole-voxrl")
    print("🎯 Project found")
except Exception as e:
    print("❌ Failed to load project:", e)
