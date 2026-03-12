"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, X, ImageIcon, ArrowRight, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface ImageUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (beforeImage: File, afterImage: File) => Promise<void>;
}

interface UploadZoneProps {
  label: string;
  file: File | null;
  preview: string | null;
  onFileSelect: (file: File) => void;
  onClear: () => void;
}

function UploadZone({ label, file, preview, onFileSelect, onClear }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("image/")) {
      onFileSelect(droppedFile);
    }
  }, [onFileSelect]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      onFileSelect(selectedFile);
    }
  };

  return (
    <div className="flex-1">
      <label className="block text-sm font-medium text-foreground mb-2">{label}</label>
      {preview ? (
        <div className="relative aspect-video rounded-lg overflow-hidden bg-[#1a1f24] border border-[#2a3038]">
          <img src={preview} alt={label} className="w-full h-full object-cover" />
          <button
            onClick={onClear}
            className="absolute top-2 right-2 p-1.5 rounded-full bg-black/60 hover:bg-black/80 transition-colors"
            aria-label="Remove image"
          >
            <X className="w-4 h-4 text-white" />
          </button>
          <div className="absolute bottom-2 left-2 right-2 bg-black/60 rounded px-2 py-1">
            <p className="text-xs text-white truncate">{file?.name}</p>
          </div>
        </div>
      ) : (
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "aspect-video rounded-lg border-2 border-dashed transition-all cursor-pointer",
            "flex flex-col items-center justify-center gap-3",
            isDragging
              ? "border-[#3a6a7a] bg-[#1a3a4a]/30"
              : "border-[#2a3038] bg-[#0d1117] hover:border-[#3a4a58] hover:bg-[#1a1f24]"
          )}
        >
          <div className="p-3 rounded-full bg-[#1a1f24] border border-[#2a3038]">
            <ImageIcon className="w-6 h-6 text-muted-foreground" />
          </div>
          <div className="text-center px-4">
            <p className="text-sm text-foreground font-medium">Drop image here</p>
            <p className="text-xs text-muted-foreground mt-1">or click to browse</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      )}
    </div>
  );
}

export function ImageUploadModal({ isOpen, onClose, onSubmit }: ImageUploadModalProps) {
  const [beforeFile, setBeforeFile] = useState<File | null>(null);
  const [afterFile, setAfterFile] = useState<File | null>(null);
  const [beforePreview, setBeforePreview] = useState<string | null>(null);
  const [afterPreview, setAfterPreview] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFileSelect = (type: "before" | "after") => (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      if (type === "before") {
        setBeforeFile(file);
        setBeforePreview(result);
      } else {
        setAfterFile(file);
        setAfterPreview(result);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleClear = (type: "before" | "after") => () => {
    if (type === "before") {
      setBeforeFile(null);
      setBeforePreview(null);
    } else {
      setAfterFile(null);
      setAfterPreview(null);
    }
  };

  const handleSubmit = async () => {
    if (!beforeFile || !afterFile) return;
    setIsSubmitting(true);
    try {
      await onSubmit(beforeFile, afterFile);
      // Reset state on success
      setBeforeFile(null);
      setAfterFile(null);
      setBeforePreview(null);
      setAfterPreview(null);
      onClose();
    } catch (error) {
      console.error("Error submitting images:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!isSubmitting) {
      setBeforeFile(null);
      setAfterFile(null);
      setBeforePreview(null);
      setAfterPreview(null);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/70 backdrop-blur-sm" 
        onClick={handleClose}
        aria-hidden="true"
      />
      
      {/* Modal */}
      <div className="relative w-full max-w-3xl bg-[#0d1117] border border-[#2a3038] rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#2a3038]">
          <div>
            <h2 className="text-lg font-semibold text-foreground">Compare Images</h2>
            <p className="text-sm text-muted-foreground mt-0.5">
              Upload before and after disaster imagery for AI analysis
            </p>
          </div>
          <button
            onClick={handleClose}
            disabled={isSubmitting}
            className="p-2 rounded-lg hover:bg-[#1a1f24] transition-colors disabled:opacity-50"
            aria-label="Close modal"
          >
            <X className="w-5 h-5 text-muted-foreground" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          <div className="flex gap-4 items-stretch">
            <UploadZone
              label="Before Disaster"
              file={beforeFile}
              preview={beforePreview}
              onFileSelect={handleFileSelect("before")}
              onClear={handleClear("before")}
            />
            
            <div className="flex items-center justify-center">
              <div className="p-2 rounded-full bg-[#1a1f24] border border-[#2a3038]">
                <ArrowRight className="w-5 h-5 text-muted-foreground" />
              </div>
            </div>
            
            <UploadZone
              label="After Disaster"
              file={afterFile}
              preview={afterPreview}
              onFileSelect={handleFileSelect("after")}
              onClear={handleClear("after")}
            />
          </div>

          {/* Info */}
          <div className="mt-4 p-3 rounded-lg bg-[#1a3a4a]/20 border border-[#2a5a6a]/30">
            <p className="text-xs text-[#7ab8d1]">
              <span className="font-medium">Tip:</span> For best results, use satellite or aerial imagery with similar framing and resolution.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-[#2a3038] bg-[#0a0d10]">
          <button
            onClick={handleClose}
            disabled={isSubmitting}
            className="px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!beforeFile || !afterFile || isSubmitting}
            className={cn(
              "flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-all",
              beforeFile && afterFile && !isSubmitting
                ? "bg-[#1a3a4a] text-white hover:bg-[#1f4557] border border-[#2a5a6a]/50"
                : "bg-[#1a1f24] text-muted-foreground border border-[#2a3038]/50 cursor-not-allowed"
            )}
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Analyze Damage
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
