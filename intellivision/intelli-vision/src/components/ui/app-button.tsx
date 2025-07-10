import React from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface AppButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  color?: "primary" | "secondary" | "tertiary";
  className?: string;
  disabled?: boolean;
}

const AppButton = React.forwardRef<HTMLButtonElement, AppButtonProps>(
  ({ children, color = "primary", className, disabled, ...props }, ref) => {
    const getColorStyles = () => {
      switch (color) {
        case "primary":
          return "bg-green-500/60 backdrop-blur-lg text-white shadow-lg hover:bg-green-500/70 border border-green-400";
        case "secondary":
          return "bg-cyan-500/60 backdrop-blur-lg text-white shadow-lg hover:bg-cyan-500/70 border border-cyan-400";
        case "tertiary":
          return "bg-white/5 hover:bg-white/15 text-white border border-white/20 shadow backdrop-blur-xl";
        default:
          return "bg-green-500/60 backdrop-blur-lg text-white shadow-lg hover:bg-green-500/70 border border-green-400";
      }
    };

    const baseStyles =
      "h-12 text-base font-extrabold button-hover rounded-3xl transition-all px-6";
    const colorStyles = getColorStyles();
    const disabledStyles = disabled
      ? "disabled:opacity-50 disabled:cursor-not-allowed"
      : "";

    return (
      <Button
        ref={ref}
        className={cn(baseStyles, colorStyles, disabledStyles, className)}
        disabled={disabled}
        {...props}
      >
        {children}
      </Button>
    );
  }
);

AppButton.displayName = "AppButton";

export default AppButton;
