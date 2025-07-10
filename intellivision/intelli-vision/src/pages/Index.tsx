import React from "react";
import { Link } from "react-router-dom";
import {
  Card,
  FeatureCard,
  AppButton,
  Highlight,
  Page,
  OuterContainer,
  InnerContainer,
} from "@/components/common/Common";
import {
  Eye,
  Play,
  BarChart3,
  Shield,
  Zap,
  Users,
  Star,
  CheckCircle,
  Camera,
  Car,
  Utensils,
  Bug,
} from "lucide-react";
import Navbar from "@/components/layout/Navbar";

/**
 * Landing page for IntelliVision.
 * Features: Hero, feature grid, benefits, and call-to-action sections.
 */
const Index = () => {
  return (
    // Main page container with animated background and navbar
    <Page>
      {/* Top navigation bar */}
      <Navbar mode="public" />

      {/* Main content wrapper */}
      <OuterContainer>
        {/* Hero Section */}
        <InnerContainer>
          <Card>
            <div className="text-center fade-in-up flex flex-col p-4 gap-6">
              {/* Highlight badge */}
              <Highlight>
                <Star className="w-5 h-5 text-yellow-400 animate-pulse" />
                <span className="text-white/90 font-medium">
                  AI-Powered Computer Vision
                </span>
              </Highlight>

              {/* Main headline */}
              <h3 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-white mb-8">
                Welcome to{" "}
                <span className="text-gradient bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent">
                  IntelliVision
                </span>
              </h3>

              {/* Subheading/description */}
              <h2 className="text-lg md:text-xl text-white/80 mb-12 max-w-5xl mx-auto leading-relaxed font-normal">
                Transform your surveillance footage into actionable insights
                with our cutting-edge AI technology. Monitor crowds, detect
                emergencies, and analyze behavior patterns in real-time.
                IntelliVision empowers organizations to automate security,
                optimize operations, and gain deep visibility into their
                environments.
              </h2>

              {/* Call-to-action buttons */}
              <div className="flex flex-col sm:flex-row gap-8 justify-center items-center w-full max-w-3xl mx-auto">
                <Link
                  to="/register"
                  className="transition-transform duration-300 hover:scale-105 w-full"
                >
                  <AppButton
                    color="primary"
                    className="w-full px-8 py-4 text-xl"
                  >
                    Get Started
                  </AppButton>
                </Link>
                <Link
                  to="/login"
                  className="transition-transform duration-300 hover:scale-105 w-full"
                >
                  <AppButton
                    color="tertiary"
                    className="w-full px-8 py-4 text-xl"
                  >
                    Sign In
                  </AppButton>
                </Link>
              </div>
            </div>
          </Card>

          {/* Main Features Grid */}
          <div className="grid grid-cols-3 mb-6 w-full mx-auto">
            {/* Security & Safety Analysis Card */}
            <Card>
              <div className="p-8">
                <div className="bg-blue-500/20 p-3 rounded-2xl w-14 h-14 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                  <Shield className="w-7 h-7 text-blue-400" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  Security & Safety Analysis
                </h3>
                <p className="text-white/80 leading-relaxed mb-6 font-normal">
                  Comprehensive security monitoring and safety analysis for
                  public spaces, facilities, and events.
                </p>
                {/* Feature list */}
                <div className="space-y-4">
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Users className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        People Counting
                      </h4>
                      <p className="text-white/70 text-sm">
                        Real-time occupancy monitoring and crowd management
                      </p>
                    </div>
                  </FeatureCard>
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Shield className="w-5 h-5 text-red-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        Emergency Detection
                      </h4>
                      <p className="text-white/70 text-sm">
                        Automatic emergency scenario identification and alerts
                      </p>
                    </div>
                  </FeatureCard>
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Camera className="w-5 h-5 text-purple-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        License Plate Recognition
                      </h4>
                      <p className="text-white/70 text-sm">
                        Vehicle identification and access control
                      </p>
                    </div>
                  </FeatureCard>
                </div>
              </div>
            </Card>

            {/* Specialized Analysis Card */}
            <Card>
              <div className="p-8">
                <div className="bg-green-500/20 p-3 rounded-2xl w-14 h-14 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                  <BarChart3 className="w-7 h-7 text-green-400" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  Specialized Analysis
                </h3>
                <p className="text-white/80 leading-relaxed mb-6">
                  Advanced AI analysis for infrastructure, agriculture, and
                  specialized monitoring applications.
                </p>
                <div className="space-y-4">
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Car className="w-5 h-5 text-blue-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        Vehicle & Infrastructure
                      </h4>
                      <p className="text-white/70 text-sm">
                        Car counting, parking analysis, and pothole detection
                      </p>
                    </div>
                  </FeatureCard>
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Utensils className="w-5 h-5 text-orange-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        Food Waste Analysis
                      </h4>
                      <p className="text-white/70 text-sm">
                        Detection and estimation of food waste for
                        sustainability
                      </p>
                    </div>
                  </FeatureCard>
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Bug className="w-5 h-5 text-yellow-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        Pest Detection
                      </h4>
                      <p className="text-white/70 text-sm">
                        Agricultural pest monitoring and identification
                      </p>
                    </div>
                  </FeatureCard>
                </div>
              </div>
            </Card>

            {/* Hotel & Hospitality Analysis Card */}
            <Card>
              <div className="p-8">
                <div className="bg-purple-500/20 p-3 rounded-2xl w-14 h-14 flex items-center justify-center mb-6 transition-transform duration-300 hover:scale-110">
                  <Eye className="w-7 h-7 text-purple-400" />
                </div>
                <h3 className="text-2xl font-bold text-white mb-4">
                  Hotel & Hospitality Analysis
                </h3>
                <p className="text-white/80 leading-relaxed mb-6">
                  Tailored AI solutions for hotels and hospitality, enhancing
                  guest experience and operational efficiency.
                </p>
                <div className="space-y-4">
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        Room Readiness
                      </h4>
                      <p className="text-white/70 text-sm">
                        Automated inspection for clean and prepared rooms
                      </p>
                    </div>
                  </FeatureCard>
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Users className="w-5 h-5 text-cyan-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        Lobby Crowd Detection
                      </h4>
                      <p className="text-white/70 text-sm">
                        Monitor lobby congestion and guest flow in real time
                      </p>
                    </div>
                  </FeatureCard>
                  <FeatureCard className="flex items-center gap-4 p-4">
                    <Star className="w-5 h-5 text-yellow-400 flex-shrink-0" />
                    <div>
                      <h4 className="text-white font-semibold">
                        Wildlife Detection
                      </h4>
                      <p className="text-white/70 text-sm">
                        Identify and alert for wildlife presence on property
                      </p>
                    </div>
                  </FeatureCard>
                </div>
              </div>
            </Card>
          </div>

          {/* Key Benefits Section */}
          <Card>
            <div className="text-center py-12 fade-in-up">
              <h2 className="text-3xl font-bold text-white mb-4">
                Why Choose IntelliVision?
              </h2>
              <p className="text-lg text-white/80 mb-8 max-w-2xl mx-auto">
                Our advanced AI technology delivers unparalleled accuracy and
                performance for all your computer vision needs.
              </p>
              <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-8 w-full mx-auto">
                <FeatureCard className="p-8">
                  <Zap className="w-10 h-10 text-yellow-400 mb-4 mx-auto transition-transform duration-300 hover:scale-110" />
                  <h3 className="text-xl font-bold text-white mb-3">
                    Lightning Fast
                  </h3>
                  <p className="text-white/80">
                    Process hours of footage in minutes with our optimized AI
                    algorithms and cloud infrastructure.
                  </p>
                </FeatureCard>
                <FeatureCard className="p-8">
                  <Shield className="w-10 h-10 text-blue-400 mb-4 mx-auto transition-transform duration-300 hover:scale-110" />
                  <h3 className="text-xl font-bold text-white mb-3">
                    Enterprise Security
                  </h3>
                  <p className="text-white/80">
                    Bank-grade encryption and compliance with industry standards
                    ensure your data stays protected.
                  </p>
                </FeatureCard>
                <FeatureCard className="p-8">
                  <BarChart3 className="w-10 h-10 text-green-400 mb-4 mx-auto transition-transform duration-300 hover:scale-110" />
                  <h3 className="text-xl font-bold text-white mb-3">
                    Actionable Insights
                  </h3>
                  <p className="text-white/80">
                    Instantly turn video into real-time, actionable intelligence
                    to drive better decisions and faster response.
                  </p>
                </FeatureCard>
              </div>
            </div>
          </Card>

          {/* Call to Action Section */}
          <Card>
            <div className="text-center py-12 fade-in-up">
              <h2 className="text-3xl font-bold text-white mb-6">
                Ready to Transform Your Security?
              </h2>
              <p className="text-lg text-white/80 mb-8 max-w-2xl mx-auto">
                Join thousands of organizations already using IntelliVision to
                enhance their security and operational efficiency.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link
                  to="/register"
                  className="transition-transform duration-300 hover:scale-105"
                >
                  <AppButton color="primary" className="px-8 py-4 text-lg">
                    Join IntelliVision
                  </AppButton>
                </Link>
                <Link
                  to="/login"
                  className="transition-transform duration-300 hover:scale-105"
                >
                  <AppButton color="secondary" className="px-8 py-4 text-lg">
                    Login
                    <Play className="w-5 h-5 ml-2" />
                  </AppButton>
                </Link>
              </div>
            </div>
          </Card>
        </InnerContainer>
      </OuterContainer>
    </Page>
  );
};

export default Index;
