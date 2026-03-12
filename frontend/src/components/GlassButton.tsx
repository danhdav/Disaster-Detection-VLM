import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary";

interface GlassButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: React.ReactNode;
}

export function GlassButton({
  variant = "primary",
  className = "",
  children,
  ...props
}: GlassButtonProps) {
  const variantClass = variant === "primary" ? "glass-btn--primary" : "glass-btn--secondary";
  return (
    <button
      type="button"
      className={`glass-btn ${variantClass} ${className}`.trim()}
      {...props}
    >
      {children}
    </button>
  );
}
