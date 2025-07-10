import React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Eye, Plus, ChevronDown, LogOut, Hotel } from "lucide-react";
import AppButton from "@/components/ui/app-button";
import { logout } from "@/utils/authFetch";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const GENERAL_TASKS = [
  { key: "people_count", label: "People Counting" },
  { key: "car_count", label: "Number Plate Detection" },
  { key: "parking_analysis", label: "Parking Analysis" },
  { key: "in_out", label: "In/Out Counting" },
  { key: "pothole_detection", label: "Pothole Detection" },
  { key: "food_waste_estimation", label: "Food Waste Estimation" },
  { key: "pest_monitoring", label: "Pest Monitoring" },
];
const HOTEL_TASKS = [
  { key: "room_readiness", label: "Room Readiness Analysis" },
  { key: "lobby_crowd_detection", label: "Lobby Crowd Detection" },
  { key: "wildlife_detection", label: "Wildlife Detection" },
];

/**
 * Navbar component renders the top navigation bar for the application.
 * Shows branding and navigation links, and conditionally renders actions based on mode.
 * Uses AppButton for consistent button styling.
 * @param {"public" | "dashboard"} mode - Determines which actions to show
 */
const Navbar = ({ mode = "public" }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const isAuthPage =
    location.pathname === "/login" || location.pathname === "/register";

  // Dashboard: handle new analysis selection
  const handleTaskSelect = (taskKey) => {
    navigate(`/upload?type=${taskKey}`);
  };
  // Dashboard: handle logout
  const handleLogout = () => {
    logout();
  };

  return (
    <header className="glass-navbar fixed top-0 left-0 right-0 z-50 fade-in-up transition-all duration-500">
      <div className="max-w-8xl mx-auto px-4 sm:px-6 lg:px-20">
        <div className="flex justify-between items-center h-16">
          {/* Branding: Logo and App Name */}
          <Link to="/" className="flex items-center fade-in-up group">
            <div className="bg-white/10 p-2 rounded-2xl mr-3 shadow-2xl border border-white/20 transition-all duration-400 group-hover:scale-110 group-hover:shadow-3xl">
              <Eye className="w-6 h-6 text-white drop-shadow-lg transition-transform duration-300 group-hover:rotate-12" />
            </div>
            <h1 className="text-2xl font-extrabold text-white tracking-tight drop-shadow-sm transition-all duration-300 group-hover:text-cyan-200 group-hover:bg-clip-text">
              IntelliVision
            </h1>
          </Link>

          {/* Actions: Public or Dashboard */}
          {mode === "dashboard" ? (
            <div
              className="flex items-center space-x-4 fade-in-up"
              style={{ animationDelay: "0.2s" }}
            >
              {/* New Analysis Dropdown */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <AppButton color="tertiary">
                    <Plus className="w-5 h-5 mr-2" />
                    New Analysis
                    <ChevronDown className="w-4 h-4 ml-2" />
                  </AppButton>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="glass-intense backdrop-blur-3xl border border-white/20 shadow-2xl rounded-2xl py-3 z-50 min-w-[280px] ring-1 ring-white/20">
                  {/* General Jobs section */}
                  <div className="px-3 py-2 text-xs font-semibold text-white/60 uppercase tracking-wider">
                    General Jobs
                  </div>
                  {GENERAL_TASKS.map((task) => (
                    <DropdownMenuItem
                      key={task.key}
                      onClick={() => handleTaskSelect(task.key)}
                      className="px-6 py-3 rounded-xl hover:bg-white/15 hover:backdrop-blur-xl font-semibold text-white/90 transition-all duration-200 cursor-pointer focus:bg-white/15 mx-2"
                    >
                      {task.label}
                    </DropdownMenuItem>
                  ))}
                  {/* Hotel Jobs section */}
                  <div className="px-3 py-2 text-xs font-semibold text-white/60 uppercase tracking-wider mt-2 border-t border-white/10">
                    Hotel Jobs
                  </div>
                  {HOTEL_TASKS.map((task) => (
                    <DropdownMenuItem
                      key={task.key}
                      onClick={() => handleTaskSelect(task.key)}
                      className="px-6 py-3 rounded-xl hover:bg-white/15 hover:backdrop-blur-xl font-semibold text-white/90 transition-all duration-200 cursor-pointer focus:bg-white/15 mx-2"
                    >
                      <Hotel className="w-4 h-4 mr-2 text-white/70" />
                      {task.label}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
              {/* Sign Out button */}
              <AppButton onClick={handleLogout} color="tertiary">
                <LogOut className="w-4 h-4 mr-2" />
                Sign Out
              </AppButton>
            </div>
          ) : (
            !isAuthPage && (
              <div
                className="flex items-center space-x-4 fade-in-up"
                style={{ animationDelay: "0.2s" }}
              >
                {/* Sign In Button */}
                <Link
                  to="/login"
                  className="transition-transform duration-300 hover:scale-105"
                >
                  <AppButton color="tertiary" className="px-6">
                    Sign In
                  </AppButton>
                </Link>
                {/* Get Started Button */}
                <Link
                  to="/register"
                  className="transition-transform duration-300 hover:scale-105"
                >
                  <AppButton color="primary">Get Started</AppButton>
                </Link>
              </div>
            )
          )}
        </div>
      </div>
    </header>
  );
};

export default Navbar;
