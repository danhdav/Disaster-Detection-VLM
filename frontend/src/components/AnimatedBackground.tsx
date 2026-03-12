"use client";

export function AnimatedBackground() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {/* Animated grid pattern */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1a1f24_1px,transparent_1px),linear-gradient(to_bottom,#1a1f24_1px,transparent_1px)] bg-[size:60px_60px] opacity-30" />
      
      {/* Floating particles */}
      <div className="absolute inset-0">
        {/* Large slow-moving orb */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[#1a3a4a]/20 rounded-full blur-3xl animate-float-slow" />
        
        {/* Medium orb */}
        <div className="absolute bottom-1/3 right-1/4 w-72 h-72 bg-[#0d2030]/30 rounded-full blur-3xl animate-float-medium" />
        
        {/* Small accent orbs */}
        <div className="absolute top-1/2 right-1/3 w-48 h-48 bg-[#1a3a4a]/15 rounded-full blur-2xl animate-float-fast" />
        <div className="absolute bottom-1/4 left-1/3 w-32 h-32 bg-[#2a5a6a]/10 rounded-full blur-2xl animate-float-reverse" />
      </div>

      {/* Scanning line effect */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute w-full h-px bg-gradient-to-r from-transparent via-[#2a5a6a]/40 to-transparent animate-scan" />
      </div>

      {/* Subtle noise texture overlay */}
      <div className="absolute inset-0 opacity-[0.015]" style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 256 256\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noise\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.65\' numOctaves=\'3\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noise)\'/%3E%3C/svg%3E")' }} />
    </div>
  );
}
