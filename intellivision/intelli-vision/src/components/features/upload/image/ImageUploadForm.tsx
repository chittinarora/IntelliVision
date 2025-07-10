import React, { useRef, useState } from "react";
import { CardContent } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";
import ImageDropzone from "@/components/features/upload/image/ImageDropzone";
import ImagePreview from "@/components/features/upload/image/ImagePreview";
import { authFetch } from "@/utils/authFetch";

interface ImageUploadFormProps {
  onUploadSuccess: () => void;
  analysisType?: string | null;
}

interface JobResponse {
  job_id: number;
  status: "pending" | "processing" | "done" | "failed";
  created_at: string;
  job_type: string;
  results?: any;
}

interface PotholeDetectionResponse {
  total_potholes: number;
  potholes: Array<{
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
    class: string;
  }>;
  output_path: string;
}

const ImageUploadForm: React.FC<ImageUploadFormProps> = ({
  onUploadSuccess,
  analysisType,
}) => {
  const [files, setFiles] = useState<File[]>([]);
  const [imagePreviewUrls, setImagePreviewUrls] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [jobId, setJobId] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (selectedFiles: FileList) => {
    const validFiles: File[] = [];
    const urls: string[] = [];

    Array.from(selectedFiles).forEach((file) => {
      if (file.type.startsWith("image/")) {
        // Check if file already exists
        const fileExists = files.some(
          (existingFile) =>
            existingFile.name === file.name && existingFile.size === file.size
        );

        if (!fileExists) {
          validFiles.push(file);
          urls.push(URL.createObjectURL(file));
        }
      }
    });

    if (validFiles.length > 0) {
      // Add to existing files instead of replacing
      setFiles((prev) => [...prev, ...validFiles]);
      setImagePreviewUrls((prev) => [...prev, ...urls]);
      setSubmitted(false);
    } else if (selectedFiles.length > 0) {
      toast({
        title: "Invalid or Duplicate Files",
        description:
          "Please select new image files (JPEG/PNG) that haven't been added yet.",
        variant: "destructive",
      });
    }
  };

  const removeImage = (index: number) => {
    // Revoke the URL to prevent memory leaks
    URL.revokeObjectURL(imagePreviewUrls[index]);

    setFiles((prev) => prev.filter((_, i) => i !== index));
    setImagePreviewUrls((prev) => prev.filter((_, i) => i !== index));
    setSubmitted(false);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      handleFileSelect(selectedFiles);
    }
    // Reset the input value so the same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleAddMoreImages = () => {
    fileInputRef.current?.click();
  };

  const resetUpload = () => {
    setFiles([]);
    setImagePreviewUrls([]);
    setSubmitted(false);
    setJobId(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    imagePreviewUrls.forEach((url) => URL.revokeObjectURL(url));
  };

  const getUploadEndpoint = () => {
    switch (analysisType) {
      case "pothole_detection":
        return "/pothole-detection/image/";
      case "food_waste_estimation":
        return "/food-waste-estimation/";
      case "car_count":
        return "/car-count/";
      case "parking_analysis":
        return "/parking-analysis/";
      case "pest_monitoring":
        return "/pest-monitoring/image/";
      case "room_readiness":
        return "/room-readiness/";
      default:
        return "/food-waste-estimation/";
    }
  };

  const submitImages = async () => {
    if (files.length === 0) return;

    setUploading(true);

    try {
      // Handle room readiness with specific endpoint (now supports multiple images)
      if (analysisType === "room_readiness") {
        const formData = new FormData();

        // Use 'images' for multiple files or 'image' for single file
        if (files.length === 1) {
          formData.append("image", files[0]);
        } else {
          files.forEach((file) => {
            formData.append("images", file);
          });
        }

        const response = await authFetch("/room-readiness/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result: JobResponse = await response.json();
          console.log("Room readiness upload successful:", result);

          setJobId(result.job_id);
          setSubmitted(true);

          toast({
            title: "Upload Complete!",
            description: `Successfully uploaded ${files.length} image${
              files.length !== 1 ? "s" : ""
            } for room readiness analysis. Processing started.`,
          });

          onUploadSuccess();
        } else {
          const errorText = await response.text();
          console.error("Room readiness upload failed:", errorText);

          toast({
            title: "Upload Failed",
            description:
              "Server error during room readiness upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      // Handle pothole detection with specific endpoint (single image only)
      if (analysisType === "pothole_detection") {
        if (files.length > 1) {
          toast({
            title: "Multiple Images Not Supported",
            description:
              "Pothole detection currently supports one image at a time. Please select a single image.",
            variant: "destructive",
          });
          setUploading(false);
          return;
        }

        const formData = new FormData();
        formData.append("image", files[0]);

        const response = await authFetch("/pothole-detection/image/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result: PotholeDetectionResponse = await response.json();
          console.log("Pothole detection successful:", result);

          setSubmitted(true);

          toast({
            title: "Analysis Complete!",
            description: `Found ${result.total_potholes} pothole${
              result.total_potholes !== 1 ? "s" : ""
            } in the image.`,
          });

          onUploadSuccess();
        } else {
          const errorText = await response.text();
          console.error("Pothole detection failed:", errorText);

          toast({
            title: "Analysis Failed",
            description:
              "Server error during pothole detection. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      // Handle pest monitoring with specific endpoint (single image only)
      if (
        analysisType === "pest_detection" ||
        analysisType === "pest_monitoring"
      ) {
        if (files.length > 1) {
          toast({
            title: "Multiple Images Not Supported",
            description:
              "Pest monitoring currently supports one image at a time. Please select a single image.",
            variant: "destructive",
          });
          setUploading(false);
          return;
        }

        const formData = new FormData();
        formData.append("image", files[0]);

        const response = await authFetch("/pest-monitoring/image/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result: JobResponse = await response.json();
          console.log("Pest monitoring upload successful:", result);

          setJobId(result.job_id);
          setSubmitted(true);

          toast({
            title: "Upload Complete!",
            description:
              "Successfully uploaded image for pest monitoring analysis. Processing started.",
          });

          onUploadSuccess();
        } else {
          const errorText = await response.text();
          console.error("Pest monitoring upload failed:", errorText);

          toast({
            title: "Upload Failed",
            description:
              "Server error during pest monitoring upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      // Handle other job types with existing logic
      const formData = new FormData();

      // Use 'images' for multiple files or 'image' for single file as per API spec
      if (files.length === 1) {
        formData.append("image", files[0]);
      } else {
        files.forEach((file) => {
          formData.append("images", file);
        });
      }

      // Add analysis type if provided
      if (analysisType) {
        formData.append("analysis_type", analysisType);
      }

      console.log(
        `Uploading ${files.length} image${files.length !== 1 ? "s" : ""} for ${
          analysisType || "general"
        } analysis:`,
        files.map((f) => f.name)
      );

      const endpoint = getUploadEndpoint();
      const response = await authFetch(endpoint, {
        method: "POST",
        requireAuth: true,
        body: formData,
        // Don't set Content-Type - let browser handle it for FormData
      });

      if (response.ok) {
        const result: JobResponse = await response.json();
        console.log("Upload successful:", result);

        setJobId(result.job_id);
        setSubmitted(true);

        toast({
          title: "Upload Complete!",
          description: `Successfully uploaded ${files.length} image${
            files.length !== 1 ? "s" : ""
          } for analysis. Processing started.`,
        });

        onUploadSuccess();
      } else {
        const errorText = await response.text();
        console.error("Upload failed:", errorText);

        let errorMessage =
          "Server error, please check your image files and try again.";

        if (errorText.includes("No image file provided")) {
          errorMessage = "No image files provided. Please select valid images.";
        }

        toast({
          title: "Upload Failed",
          description: errorMessage,
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error("Image upload error:", error);

      if (error instanceof Error && error.message.includes("Authentication")) {
        toast({
          title: "Authentication Required",
          description: "Please log in to upload images.",
          variant: "destructive",
        });
        return;
      }

      toast({
        title: "Network Error",
        description:
          "Could not reach the server. Please check your connection.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <CardContent className="p-8 rounded-b-5xl transition-all duration-300">
      <ImageDropzone
        onDrop={() => {}} // Remove drag and drop functionality
        onDragOver={() => {}} // Remove drag and drop functionality
        onClick={() => fileInputRef.current?.click()}
        hasImages={files.length > 0}
        analysisType={analysisType}
      />

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple={
          analysisType !== "pothole_detection" &&
          analysisType !== "pest_detection" &&
          analysisType !== "pest_monitoring"
        } // Remove room_readiness from single image restriction
        onChange={handleFileInput}
        className="hidden"
      />

      {files.length > 0 && (
        <ImagePreview
          files={files}
          imagePreviewUrls={imagePreviewUrls}
          uploading={uploading}
          submitted={submitted}
          submitImages={submitImages}
          resetUpload={resetUpload}
          removeImage={removeImage}
          onAddMoreImages={
            analysisType !== "pothole_detection" &&
            analysisType !== "pest_detection" &&
            analysisType !== "pest_monitoring"
              ? handleAddMoreImages
              : undefined
          } // Remove room_readiness from single image restriction
          jobIds={jobId ? [jobId] : []}
        />
      )}
    </CardContent>
  );
};

export default ImageUploadForm;
