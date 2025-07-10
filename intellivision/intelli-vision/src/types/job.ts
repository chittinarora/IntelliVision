export interface Job {
  id: number;
  status: "pending" | "processing" | "done" | "failed";
  input_video: string;
  created_at: string;
  updated_at: string;
  job_type?:
    | "people_count"
    | "people-count"
    | "car_count"
    | "car-count"
    | "in_out"
    | "parking_analysis"
    | "emergency_count"
    | "pothole_detection"
    | "food_waste_estimation"
    | "food-waste-estimation"
    | "food-waste"
    | "food_waste"
    | "pest_monitoring"
    | "pest_detection"
    | "room_readiness"
    | "lobby_crowd_detection"
    | "lobby-detection"
    | "lobby_detection"
    | "wildlife_detection";

  // New standardized fields
  output_video?: string;
  output_image?: string;
  output_url?: string;
  csv_report?: string;
  xlsx_report?: string;
  message?: string;

  results?: {
    // Common fields
    alerts?: string[];

    // New unified media handling
    media_type?: "image" | "video";

    // Add data property for nested result structures
    data?: {
      items?: Array<{
        name: string;
        estimated_portion: string | number;
        estimated_calories: number;
        tags: string[];
      }>;
      total_calories?: number;
      waste_summary?: string;
      person_count?: number;
      raw_track_count?: number;
      car_count?: number;
      vehicle_count?: number;
      plate_count?: number;
      in?: number;
      out?: number;
      fast_in_count?: number;
      fast_out_count?: number;
      pothole_count?: number;
      total_potholes?: number;
      detected_snakes?: number;
      pest_count?: number;
      zone_counts?: { [zoneName: string]: number };
      zone_thresholds?: { [zoneName: string]: number };
      occupancy_count?: number;
      crowd_level?: "low" | "medium" | "high" | "critical";
      alert_triggered?: boolean;
      wildlife_detected?: boolean;
      wildlife_count?: number;
      wildlife_types?: string[];
      risk_level?: "low" | "medium" | "high";
      checklist?: Array<{
        item: string;
        status: "present" | "missing" | "defective";
        coordinates?: { x: number; y: number };
        notes?: string;
        note?: string;
        box?: number[];
      }>;
      room_score?: number;
      readiness_score?: number;
      readiness_status?:
        | "ready"
        | "needs_attention"
        | "not_ready"
        | "Guest Ready"
        | "Not Guest Ready";
      [key: string]: any;
    };

    // People counting
    person_count?: number;
    raw_track_count?: number;

    // Car counting
    plates_detected?: string[];
    plate_detection_times?: { [key: string]: string };
    plate_count?: number;
    vehicle_count?: number;
    car_count?: number;
    annotated_video_path?: string;
    csv_report_path?: string;
    summary?: string;
    history?: any;

    // In/out counting
    in?: number;
    out?: number;

    // Emergency counting
    unique_people?: number;
    total_crossings?: number;
    final_counts?: {
      line1: { in: number; out: number };
      line2: { in: number; out: number };
    };

    // Pothole detection
    pothole_count?: number;
    total_potholes?: number;
    pothole_locations?: [number, number][];
    frames?: Array<{
      frame_index: number;
      potholes: Array<{
        x: number;
        y: number;
        width: number;
        height: number;
        confidence: number;
        class: string;
      }>;
    }>;
    potholes?: Array<{
      x: number;
      y: number;
      width: number;
      height: number;
      confidence: number;
      class: string;
    }>;
    output_path?: string; // Legacy - kept for backward compatibility

    // Food waste estimation
    items?: Array<{
      name: string;
      estimated_portion: string | number;
      estimated_calories: number;
      tags: string[];
    }>;
    total_calories?: number;
    waste_summary?: string;

    // Pest monitoring
    pest_count?: number;
    detected_snakes?: number;
    mongo_id?: string;
    types?: string[];

    // Room readiness
    zone?: string;
    checklist?: Array<{
      item: string;
      status: "present" | "missing" | "defective";
      coordinates?: { x: number; y: number };
      notes?: string;
      note?: string;
      box?: number[];
    }>;
    room_score?: number;
    readiness_score?: number;
    readiness_status?:
      | "ready"
      | "needs_attention"
      | "not_ready"
      | "Guest Ready"
      | "Not Guest Ready";
    status?: string;
    cleanliness_issues?: string[];
    fail_reasons?: string[];
    instructions?: string[] | string;
    error?: string;
    image_width?: number;
    image_height?: number;

    // Lobby crowd detection
    zone_counts?: { [zoneName: string]: number };
    occupancy_count?: number;
    crowd_level?: "low" | "medium" | "high" | "critical";
    alert_triggered?: boolean;
    output_video_path?: string;

    // Wildlife detection
    wildlife_detected?: boolean;
    wildlife_count?: number;
    wildlife_types?: string[];
    risk_level?: "low" | "medium" | "high";
  };

  meta?: {
    fps?: number;
    processed_at?: string;
    detection_strategies?: string[];
    depth_method?: string;
    [key: string]: any;
  };

  error?: string;

  // Legacy fields (deprecated but kept for backward compatibility)
  person_count?: number;
  type?: "people_count" | "emergency" | "number_plate";
  plate_count?: number;
}
