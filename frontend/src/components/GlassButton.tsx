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
  const base =
    "px-6 py-3 rounded-full font-medium transition-all duration-200 text-sm";
  const variantClass =
    variant === "primary"
      ? "bg-[#1a3a4a] text-white hover:bg-[#1f4557] border border-[#2a5a6a]/50"
      : "bg-[#1a1f24] text-white hover:bg-[#252b32] border border-[#2a3038]/50";
  return (
    <button
      type="button"
      className={[base, variantClass, className].filter(Boolean).join(" ")}
      {...props}
    >
      {children}
    </button>
  );
}
