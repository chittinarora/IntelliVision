import * as React from "react";
import { Button, buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/**
 * Card primitive for consistent card styling across the app.
 * Usage: Use <Card> for all card-like containers. Do not use raw divs with glass-card classes.
 * Example:
 *   <Card>Content here</Card>
 */
export function Card({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`glass-card rounded-4xl p-6 shadow-2xl m-4 $ {className}`}
      {...props}
    />
  );
}

/**
 * CardContent primitive for card content area with standard padding.
 * Usage: Use <CardContent>...</CardContent> inside Card or FeatureCard.
 */
export function CardContent({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`flex flex-col p-4 gap-6 text-center fade-in-up ${className}`}
      {...props}
    />
  );
}

/**
 * CardTitle primitive for card/section titles.
 * Usage: Use <CardTitle>Section Title</CardTitle> inside Card, Section, or FeatureCard.
 */
export function CardTitle({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={`text-white md:text-5xl lg:text-6xl font-extrabold ${className}`}
      {...props}
    />
  );
}

/**
 * CardSubtitle primitive for subtitles or secondary headings.
 * Usage: Use <CardSubtitle>Subtitle</CardSubtitle> for subtitles.
 */
export function CardSubtitle({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h4
      className={`text-lg font-semibold text-white/60 mx-auto leading-relaxed ${className}`}
      {...props}
    />
  );
}

export function TextGradient({
  className = "",
  children,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={`text-gradient bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400 bg-clip-text text-transparent ${className}`}
      {...props}
    >
      {children}
    </span>
  );
}

export function Page({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  return (
    <div
      className={`min-h-screen bg-navy-gradient relative overflow-hidden page-enter page-enter-active ${className}`}
      {...props}
    />
  );
}

export function OuterContainer({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  return (
    <div
      className={`py-20 px-4 sm:px-6 lg:px-20 xl:px-24 max-w-screen-2xl mx-auto ${className}`}
      {...props}
    />
  );
}

export function InnerContainer({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  return (
    <div className={`max-w-screen-2xl mx-auto my-6 ${className}`} {...props} />
  );
}

export function Highlight({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLElement>) {
  return (
    <span
      className={`self-center inline-flex items-center space-x-3 glass-card rounded-full px-6 py-3 transition-all duration-500 hover:scale-105 ${className}`}
      {...props}
    />
  );
}

/**
 * FeatureCard primitive for feature list items or highlights.
 * Usage: Use <FeatureCard> for feature highlights or list items.
 * Example:
 *   <FeatureCard>Feature content</FeatureCard>
 */
export function FeatureCard({
  className = "",
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={`feature-card ${className}`} {...props} />;
}

/**
 * AppButton primitive for all button usage.
 * Usage: Use <AppButton color="primary|secondary|tertiary"> for design system buttons.
 * Example:
 *   <AppButton color="primary">Save</AppButton>
 */
export interface AppButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  color?: "primary" | "secondary" | "tertiary";
  className?: string;
  disabled?: boolean;
}

export const AppButton = React.forwardRef<HTMLButtonElement, AppButtonProps>(
  ({ children, color = "primary", className, disabled, ...props }, ref) => {
    // Map color prop to design system classes
    const getColorStyles = () => {
      switch (color) {
        case "primary":
          return "bg-green-500/60 backdrop-blur-lg text-white shadow-lg hover:bg-green";
        case "secondary":
          return "bg-cyan-500/60 backdrop-blur-lg text-white shadow-lg hover:bg-cyan-500/70 border border-cyan-400/20";
        case "tertiary":
          return "bg-white/5 hover:bg-white/15 text-white border border-white/20 shadow backdrop-blur-xl";
        default:
          return "bg-green-500/60 backdrop-blur-lg text-white shadow-lg hover:bg-green-500/70 border border-green-400";
      }
    };
    const baseStyles =
      "h-12 text-base font-extrabold button-hover rounded-3xl transition-transform duration-300 hover:scale-105 px-6 py-3";
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
