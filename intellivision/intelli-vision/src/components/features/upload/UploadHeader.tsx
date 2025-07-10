
import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Eye, ArrowLeft, LogOut } from 'lucide-react';
import { logout } from '@/utils/authFetch';

const UploadHeader = () => {
  const handleLogout = () => {
    logout();
  };

  return (
    <header className="bg-white/10 backdrop-blur-3xl border-b border-white/20 shadow-2xl fixed top-0 left-0 right-0 z-50 fade-in-up">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center fade-in-up">
            <Link to="/dashboard" className="inline-flex items-center text-white/80 hover:text-white transition-all duration-300 hover:scale-105 mr-6 font-semibold tracking-wide">
              <ArrowLeft className="w-4 h-4 mr-2" />
              <span className="hidden sm:inline">Back to Dashboard</span>
            </Link>
            <div className="bg-gradient-to-r from-slate-700 to-slate-800 p-2 rounded-2xl mr-3 shadow-2xl border border-white/20">
              <Eye className="w-6 h-6 text-white drop-shadow-lg" />
            </div>
            <h1 className="text-2xl font-extrabold text-white tracking-tight drop-shadow-sm">IntelliVision</h1>
          </div>
          <div className="flex items-center space-x-4 fade-in-up" style={{animationDelay: '0.2s'}}>
            <Button 
              onClick={handleLogout}
              variant="outline" 
              className="border border-white/20 text-white bg-white/10 hover:bg-white/20 hover:text-white backdrop-blur-xl transition-all duration-300 font-semibold rounded-2xl px-6 shadow-lg button-hover"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Sign Out
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default UploadHeader;
