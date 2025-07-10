import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { ArrowLeft, Sparkles, Shield, Zap } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import WebcamCapture from "@/components/auth/WebcamCapture";
import { loginWithFace, pollTaskStatus } from "@/utils/auth/faceAuth";
import Navbar from "@/components/layout/Navbar";

const Login = () => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  /**
   * Handles face authentication process
   * @param imageBlob - Captured image from webcam
   */
  const handleImageCapture = async (imageBlob: Blob) => {
    setIsSubmitting(true);

    try {
      // Initiate face login process
      const { task_id } = await loginWithFace({
        username: "",
        image: imageBlob,
      });

      // Poll for authentication result
      const result = await pollTaskStatus(task_id);

      if (result.success && result.token) {
        // Store authentication token
        localStorage.setItem("accessToken", result.token);

        const welcomeMessage = result.name
          ? `Welcome back, ${result.name}!`
          : result.message || "Welcome back!";

        toast({
          title: "Login Successful",
          description: welcomeMessage,
          variant: "default",
        });

        // Navigate to dashboard after brief delay
        setTimeout(() => {
          navigate("/dashboard");
        }, 1000);
      } else {
        toast({
          title: "Login Failed",
          description:
            result.message || "Authentication failed. Please try again.",
          variant: "destructive",
        });
      }
    } catch (error) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Authentication failed. Please try again.";

      toast({
        title: "Login Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-navy-gradient relative overflow-hidden page-enter page-enter-active">
      <Navbar mode="public" />

      {/* Animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 left-10 w-40 h-40 bg-white/4 rounded-full blur-2xl animate-pulse floating"></div>
        <div
          className="absolute bottom-40 right-1/4 w-56 h-56 bg-cyan-400/3 rounded-full blur-3xl animate-pulse floating"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="absolute top-1/2 left-1/4 w-32 h-32 bg-blue-400/4 rounded-full blur-xl animate-pulse floating"
          style={{ animationDelay: "1s" }}
        ></div>
        <div
          className="absolute top-1/3 right-10 w-24 h-24 bg-purple-400/3 rounded-full blur-lg animate-pulse floating"
          style={{ animationDelay: "3s" }}
        ></div>
      </div>

      <div className="pt-20 px-4 sm:px-6 lg:px-20 min-h-screen flex items-center justify-center">
        <div className="w-full max-w-8xl grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Side - Welcome Content */}
          <div className="space-y-10 fade-in-up slide-in-left">
            <div className="space-y-6">
              {/* Feature badge */}
              <div className="inline-flex items-center space-x-3 glass-card rounded-full px-6 py-3 transition-all duration-500 hover:scale-105">
                <Sparkles className="w-5 h-5 text-yellow-400 animate-pulse" />
                <span className="text-white/90 font-medium">
                  Welcome Back to IntelliVision
                </span>
              </div>

              {/* Main heading */}
              <h1 className="text-5xl lg:text-6xl font-bold text-white">
                Sign In with
                <span className="block text-gradient bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent leading-relaxed">
                  Face Recognition
                </span>
              </h1>

              <p className="text-xl text-white/80 leading-relaxed max-w-lg">
                Experience the future of authentication. Secure, fast, and
                effortless access to your IntelliVision dashboard.
              </p>
            </div>

            {/* Feature highlights */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              <div className="glass-card rounded-3xl p-6 border-subtle stagger-fade">
                <Shield className="w-8 h-8 text-blue-400 mb-3 transition-transform duration-300 hover:scale-110" />
                <h3 className="text-white font-semibold mb-2">
                  Secure Authentication
                </h3>
                <p className="text-white/70 text-sm">
                  Advanced facial recognition technology keeps your account safe
                </p>
              </div>

              <div className="glass-card rounded-3xl p-6 border-subtle stagger-fade">
                <Zap className="w-8 h-8 text-green-400 mb-3 transition-transform duration-300 hover:scale-110" />
                <h3 className="text-white font-semibold mb-2">
                  Instant Access
                </h3>
                <p className="text-white/70 text-sm">
                  Get authenticated in seconds with our AI-powered system
                </p>
              </div>
            </div>
          </div>

          {/* Right Side - Login Form */}
          <div
            className="fade-in-up slide-in-right"
            style={{ animationDelay: "0.2s" }}
          >
            <Card className="content-card transition-all duration-500 hover:scale-[1.02]">
              <CardContent className="p-8">
                <div className="space-y-6">
                  <WebcamCapture
                    onCapture={handleImageCapture}
                    isCapturing={isSubmitting}
                    buttonText={
                      isSubmitting ? "Signing In..." : "Capture Photo & Sign In"
                    }
                  />
                </div>

                {/* Navigation links */}
                <div className="mt-8 text-center space-y-4">
                  <p className="text-sm text-white/80">
                    Don't have an account?{" "}
                    <Link
                      to="/register"
                      className="font-semibold text-white hover:underline transition-all duration-300 hover:text-cyan-300"
                    >
                      Sign up here
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

export default Login;
