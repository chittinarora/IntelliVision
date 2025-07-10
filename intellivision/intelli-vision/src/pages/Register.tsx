import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { ArrowLeft, Users, Shield, Zap, Star } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import WebcamCapture from "@/components/auth/WebcamCapture";
import { registerWithFace, pollTaskStatus } from "@/utils/auth/faceAuth";
import Navbar from "@/components/layout/Navbar";

const Register = () => {
  const [username, setUsername] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  /**
   * Handles face registration process
   * @param imageBlob - Captured image from webcam
   */
  const handleImageCapture = async (imageBlob: Blob) => {
    // Validate username before proceeding
    if (!username.trim()) {
      toast({
        title: "Username Required",
        description: "Please enter a username first.",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);

    try {
      // Initiate face registration process
      const { task_id } = await registerWithFace({
        username: username.trim(),
        image: imageBlob,
      });

      // Poll for registration result
      const result = await pollTaskStatus(task_id);

      if (result.success && result.token) {
        // Store authentication token
        localStorage.setItem("accessToken", result.token);

        const welcomeMessage = result.name
          ? `Welcome to IntelliVision, ${result.name}!`
          : result.message || "Account created successfully!";

        toast({
          title: "Registration Successful",
          description: welcomeMessage,
          variant: "default",
        });

        // Navigate to dashboard after brief delay
        setTimeout(() => {
          navigate("/dashboard");
        }, 1000);
      } else {
        toast({
          title: "Registration Failed",
          description:
            result.message || "Something went wrong. Please try again.",
          variant: "destructive",
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Something went wrong. Please try again.";

      toast({
        title: "Registration Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-navy-gradient relative overflow-hidden page-enter page-enter-active">
      <Navbar />

      {/* Animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 right-10 w-32 h-32 bg-white/5 rounded-full blur-xl animate-pulse floating"></div>
        <div
          className="absolute bottom-40 left-1/4 w-40 h-40 bg-white/4 rounded-full blur-2xl animate-pulse floating"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="absolute top-1/3 right-1/3 w-28 h-28 bg-white/3 rounded-full blur-lg animate-pulse floating"
          style={{ animationDelay: "1.5s" }}
        ></div>
        <div
          className="absolute top-1/4 left-10 w-36 h-36 bg-cyan-400/3 rounded-full blur-2xl animate-pulse floating"
          style={{ animationDelay: "3s" }}
        ></div>
        <div
          className="absolute bottom-1/4 right-20 w-20 h-20 bg-purple-400/4 rounded-full blur-xl animate-pulse floating"
          style={{ animationDelay: "4s" }}
        ></div>
      </div>

      <div className="pt-20 px-4 sm:px-6 lg:px-20 min-h-screen flex items-center justify-center">
        <div className="w-full max-w-8xl grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Benefits & Features */}
          <div className="space-y-8 bounce-in">
            <div className="space-y-6">
              {/* Feature badge */}
              <div className="inline-flex items-center space-x-3 glass-card rounded-full px-6 py-3 transition-all duration-500 hover:scale-105 shimmer">
                <Star className="w-5 h-5 text-yellow-400 animate-pulse" />
                <span className="text-white/90 font-medium">
                  Join IntelliVision Today
                </span>
              </div>

              {/* Main heading */}
              <h1 className="text-5xl lg:text-6xl font-bold text-white leading-tight">
                Create Your
                <span className="block text-gradient bg-gradient-to-r from-emerald-400 to-green-400 bg-clip-text text-transparent">
                  Secure Account
                </span>
              </h1>

              <p className="text-xl text-white/80 leading-relaxed max-w-full">
                Join thousands of users who trust IntelliVision for intelligent
                computer vision solutions. Get started in minutes.
              </p>
            </div>

            {/* Feature highlights */}
            <div className="space-y-4">
              <div className="flex items-start space-x-4 glass-card rounded-3xl p-6 border-subtle stagger-fade">
                <div className="bg-emerald-500/20 p-3 rounded-2xl">
                  <Users className="w-6 h-6 text-emerald-400" />
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1">
                    Join 10,000+ Users
                  </h3>
                  <p className="text-white/70 text-sm">
                    Be part of our growing community of AI enthusiasts
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4 glass-card rounded-3xl p-6 border-subtle stagger-fade">
                <div className="bg-blue-500/20 p-3 rounded-2xl">
                  <Shield className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1">
                    Advanced Security
                  </h3>
                  <p className="text-white/70 text-sm">
                    Your data is protected with enterprise-grade encryption
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4 glass-card rounded-3xl p-6 border-subtle stagger-fade">
                <div className="bg-purple-500/20 p-3 rounded-2xl">
                  <Zap className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h3 className="text-white font-semibold mb-1">
                    Lightning Fast Setup
                  </h3>
                  <p className="text-white/70 text-sm">
                    Get up and running in under 2 minutes
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Side - Registration Form */}
          <div className="bounce-in" style={{ animationDelay: "0.2s" }}>
            <Card className="content-card transition-all duration-500 hover:scale-[1.02]">
              <CardContent className="p-8">
                <div className="space-y-6">
                  {/* Username input */}
                  <div className="space-y-2">
                    <Label
                      htmlFor="username"
                      className="text-white/90 font-medium"
                    >
                      Username
                    </Label>
                    <Input
                      id="username"
                      type="text"
                      placeholder="Choose a username"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      required
                      className="h-12 glass-subtle border-white/20 text-white placeholder:text-white/60 rounded-2xl focus:bg-white/20 transition-all duration-300 px-4 micro-bounce"
                    />
                  </div>

                  {/* Face capture section */}
                  <div className="space-y-2">
                    <Label className="text-white/90 font-medium">
                      Face Authentication
                    </Label>
                    <WebcamCapture
                      onCapture={handleImageCapture}
                      isCapturing={isSubmitting}
                      buttonText={
                        isSubmitting
                          ? "Creating Account..."
                          : "Capture Photo & Create Account"
                      }
                    />
                  </div>
                </div>

                {/* Navigation links */}
                <div className="mt-8 text-center space-y-4">
                  <p className="text-sm text-white/80">
                    Already have an account?{" "}
                    <Link
                      to="/login"
                      className="font-semibold text-white hover:underline transition-all duration-300 hover:text-cyan-300"
                    >
                      Sign in here
                    </Link>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Register;
