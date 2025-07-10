
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Frown } from "lucide-react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-navy-gradient relative overflow-hidden text-white p-4">
      {/* Animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-20 left-10 w-40 h-40 bg-white/4 rounded-full blur-2xl animate-pulse floating"></div>
        <div className="absolute bottom-40 right-1/4 w-56 h-56 bg-cyan-400/3 rounded-full blur-3xl animate-pulse floating" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-1/2 left-1/4 w-32 h-32 bg-blue-400/4 rounded-full blur-xl animate-pulse floating" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/3 right-10 w-24 h-24 bg-purple-400/3 rounded-full blur-lg animate-pulse floating" style={{ animationDelay: '3s' }}></div>
      </div>

      <div className="text-center fade-in-up">
        <div className="glass-intense backdrop-blur-3xl p-8 rounded-full w-40 h-40 mx-auto mb-8 flex items-center justify-center shadow-2xl border border-white/20 hero-glow">
          <Frown className="w-20 h-20 text-white/80" />
        </div>
        <h1 className="text-6xl font-extrabold mb-4 tracking-tight">404</h1>
        <p className="text-2xl text-white/80 mb-8 max-w-md mx-auto">Oops! The page you are looking for does not exist.</p>
        <a href="/" className="text-lg font-semibold text-white glass-card hover:bg-white/20 border border-white/20 px-8 py-3 rounded-2xl transition-all duration-300 shadow-lg button-hover">
          Return to Home
        </a>
      </div>
    </div>
  );
};

export default NotFound;
