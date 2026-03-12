import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { GlassButton } from "../../components/GlassButton";
import { ChatBot } from "../../components/ChatBot";
import { ImageUploadModal } from "../../components/ImageUploadModal";
import { AnimatedBackground } from "../../components/AnimatedBackground";
import "./-app.css";

export const Route = createFileRoute("/(base)/")({
  component: Home,
});

function Home() {
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleImageSubmit = async (beforeImage: File, afterImage: File) => {
    console.log("Submitting images for analysis:", {
      before: beforeImage.name,
      after: afterImage.name,
    });
    await new Promise((resolve) => setTimeout(resolve, 2000));
  };

  const scrollToHowItWorks = () => {
    document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-[#0a0d10] text-foreground relative">
      <AnimatedBackground />

      {/* Hero Section */}
      <main className="relative flex flex-col items-center justify-center min-h-screen px-4 pb-24">
        <div className="text-center max-w-2xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6 tracking-tight text-balance">
            Disaster Detection
          </h1>
          <p className="text-lg text-gray-400 mb-10 leading-relaxed max-w-lg mx-auto text-pretty">
            Compare before & after imagery. Assess damage with AI.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <GlassButton variant="primary" onClick={() => setIsModalOpen(true)}>
              Compare before & after
            </GlassButton>
            <GlassButton variant="secondary" onClick={scrollToHowItWorks}>
              How it works
            </GlassButton>
          </div>
        </div>
      </main>

      {/* How It Works Section */}
      <section id="how-it-works" className="relative py-24 px-4 border-t border-[#1a1f24]">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-16 text-white">
            How It Works
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-14 h-14 rounded-2xl bg-[#1a3a4a] border border-[#2a5a6a]/50 flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold text-white">1</span>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">Upload Images</h3>
              <p className="text-sm text-gray-400 leading-relaxed">
                Upload satellite or aerial imagery from before and after a disaster event.
              </p>
            </div>
            <div className="text-center">
              <div className="w-14 h-14 rounded-2xl bg-[#1a3a4a] border border-[#2a5a6a]/50 flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold text-white">2</span>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">AI Analysis</h3>
              <p className="text-sm text-gray-400 leading-relaxed">
                Our AI model compares the images to detect structural damage and changes.
              </p>
            </div>
            <div className="text-center">
              <div className="w-14 h-14 rounded-2xl bg-[#1a3a4a] border border-[#2a5a6a]/50 flex items-center justify-center mx-auto mb-4">
                <span className="text-xl font-bold text-white">3</span>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">Get Results</h3>
              <p className="text-sm text-gray-400 leading-relaxed">
                Receive detailed damage assessment reports with severity classifications.
              </p>
            </div>
          </div>
        </div>
      </section>

      <ImageUploadModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSubmit={handleImageSubmit}
      />

      <ChatBot />
    </div>
  );
}
