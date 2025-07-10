
import React, { useMemo } from "react";
import { Trash2 } from "lucide-react";
import { Job } from "@/types/job";

interface FoodWasteResultProps {
  job: Job;
  isCollapsed: boolean;
}

const FoodWasteResult: React.FC<FoodWasteResultProps> = ({
  job,
  isCollapsed,
}) => {
  const foodWasteData = useMemo(() => {
    console.log("FoodWasteResult processing job:", job);
    console.log("Job type:", job.job_type);
    console.log("Job results:", job.results);

    // Check if this is a food waste job
    const jobType = job.job_type;
    const isFoodWasteJob =
      jobType === "food_waste_estimation" ||
      jobType === "food-waste" ||
      jobType === "food-waste-estimation" ||
      jobType === "food_waste";

    if (!isFoodWasteJob || !job.results) {
      console.log("Not a food waste job or no results");
      return {
        totalCalories: 0,
        totalItems: 0,
        totalGrams: 0,
        allItems: [],
        wasteSummary: "",
      };
    }

    // Support both array and object (batch and single)
    const dataArray = Array.isArray(job.results)
      ? job.results.map((r) => r.data)
      : job.results.data
      ? [job.results.data]
      : [];

    let totalCalories = 0;
    let totalItems = 0;
    let totalGrams = 0;
    let allItems: Array<{
      name: string;
      estimated_portion: string | number;
      estimated_calories: number;
      tags?: string[];
    }> = [];
    let wasteSummary = "";

    if (dataArray.length > 0) {
      dataArray.forEach((data) => {
        if (data && Array.isArray(data.items)) {
          totalCalories += data.total_calories || 0;
          totalItems += data.items.length;
          allItems = allItems.concat(
            data.items.map((item) => ({
              name: item.name,
              estimated_portion: item.estimated_portion,
              estimated_calories: item.estimated_calories,
              tags: item.tags || [],
            }))
          );
          wasteSummary = data.waste_summary || wasteSummary;
          data.items.forEach((item) => {
            const portion =
              typeof item.estimated_portion === "string"
                ? parseFloat(item.estimated_portion.replace(/[^\d.]/g, "")) || 0
                : item.estimated_portion || 0;
            totalGrams += portion;
          });
        }
      });
    }

    console.log("Processed food waste data:", {
      totalCalories,
      totalItems,
      totalGrams,
      allItems: allItems.length,
    });
    return { totalCalories, totalItems, totalGrams, allItems, wasteSummary };
  }, [job.results, job.job_type]);

  if (
    !foodWasteData.totalCalories ||
    !foodWasteData.totalItems ||
    !foodWasteData.totalGrams
  ) {
    console.log("No food waste data to display");
    return null;
  }

  return (
    <div className="space-y-6 bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-6">
      <div className="flex items-center justify-between">
        <p className="flex items-center text-xl font-bold text-white">
          <Trash2 className="w-7 h-7 mr-3 text-emerald-400 flex-shrink-0" />
          Food Waste Analysis Results
        </p>
      </div>

      {/* Overall Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="text-center bg-white/5 backdrop-blur-sm rounded-2xl p-4 border border-white/10">
          <div className="text-2xl font-bold text-emerald-400">
            {foodWasteData.totalCalories}
          </div>
          <div className="text-sm text-white/70 mt-1">Total Calories</div>
        </div>
        <div className="text-center bg-white/5 backdrop-blur-sm rounded-2xl p-4 border border-white/10">
          <div className="text-2xl font-bold text-emerald-400">
            {Math.round(foodWasteData.totalGrams)}g
          </div>
          <div className="text-sm text-white/70 mt-1">Total Weight</div>
        </div>
        <div className="text-center bg-white/5 backdrop-blur-sm rounded-2xl p-4 border border-white/10">
          <div className="text-2xl font-bold text-emerald-400">
            {foodWasteData.totalItems}
          </div>
          <div className="text-sm text-white/70 mt-1">Items Detected</div>
        </div>
      </div>

      {/* Detailed sections - only show when not collapsed */}
      {!isCollapsed && (
        <>
          {/* Compact Item Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {foodWasteData.allItems.map((item, idx) => (
              <div
                key={idx}
                className="bg-white/10 rounded-xl p-3 text-white text-sm flex flex-col gap-1 border border-white/10"
              >
                <div className="font-semibold truncate">{item.name}</div>
                <div className="flex justify-between text-xs text-white/70">
                  <span>{item.estimated_portion}g</span>
                  <span>{item.estimated_calories} cal</span>
                </div>
                {item.tags && item.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-1">
                    {item.tags.map((tag, tagIdx) => (
                      <span
                        key={tagIdx}
                        className="px-2 py-0.5 bg-emerald-600/20 text-emerald-300 rounded-full text-[10px]"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Waste Summary */}
          {foodWasteData.wasteSummary && (
            <div className="text-sm text-white/80 bg-white/10 rounded-xl p-3 border border-white/10">
              "{foodWasteData.wasteSummary}"
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default FoodWasteResult;
