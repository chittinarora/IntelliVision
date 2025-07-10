import React from "react";

interface WelcomeSectionProps {
  username: string | null;
}

const WelcomeSection = ({ username }: WelcomeSectionProps) => {
  return (
    <div className="mb-16 text-center fade-in-up">
      <h2 className="text-6xl font-extrabold text-white mb-4 tracking-tight">
        Welcome back{username ? `, ${username}` : ""}
      </h2>
      <p className="text-white/70 text-xl max-w-5xl mx-auto leading-relaxed font-light">
        Explore your dashboard below.
        <br></br> Launch a new analysis, view your stats, and review past
        results to unlock powerful insights with just a few clicks.
      </p>
    </div>
  );
};

export default WelcomeSection;
