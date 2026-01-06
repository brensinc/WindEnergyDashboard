// Database types

export interface Project {
  id: string
  user_id: string
  name: string
  description: string | null
  latitude: number
  longitude: number
  hub_height_m: number
  rotor_diameter_m: number
  turbine_model: string | null
  rated_power_kw: number | null
  cached_annual_revenue: number | null
  cached_capacity_factor: number | null
  cached_iso: string | null
  cached_at: string | null
  created_at: string
  updated_at: string
}

export interface ProjectFormData {
  name: string
  description?: string
  latitude: number
  longitude: number
  hub_height_m: number
  rotor_diameter_m: number
  turbine_model?: string
  rated_power_kw?: number
}

export interface IsoPrice {
  id: number
  iso: string
  price_type: string
  timestamp: string
  price_usd_mwh: number
}

export interface Report {
  id: string
  project_id: string
  user_id: string
  report_type: string
  storage_path: string | null
  generated_at: string
  start_date: string | null
  end_date: string | null
  parameters: Record<string, unknown> | null
}

// API Response types

export interface WindDataResponse {
  timestamps: string[]
  wind_speeds: number[]
  wind_directions: number[]
  power_outputs: number[]
  capacity_factors: number[]
  metadata: {
    latitude: number
    longitude: number
    hub_height_m: number
    rotor_diameter_m: number
    rated_power_kw: number
    iso: string | null
    pricing_type: string | null
  }
}

export interface PriceDataResponse {
  timestamps: string[]
  prices: number[]
  iso: string
  price_type: string
}

export interface RevenueDataResponse {
  timestamps: string[]
  revenues: number[]
  prices: number[]
  power_outputs: number[]
  total_revenue: number
  total_energy_mwh: number
  average_price: number
  capacity_factor: number
}

// Map types

export interface MapMarker {
  id: string
  latitude: number
  longitude: number
  name: string
  type: 'project' | 'prospect'
}

// Chart types

export interface MonthlyData {
  month: string
  energy_mwh: number
  revenue: number
  capacity_factor: number
}

export interface HourlyPattern {
  hour: number
  month: number
  value: number
}

export interface WindRoseData {
  direction: string
  frequency: number
  avgSpeed: number
}

// Analytics types

export interface AnalyticsRequest {
  project_id: string
  latitude: number
  longitude: number
  hub_height_m: number
  rotor_diameter_m: number
  rated_power_kw: number
  start_date: string
  end_date: string
  pricing_mode: 'market' | 'fixed'
  fixed_price?: number
  iso_override?: string
}

export interface SummaryStats {
  total_energy_mwh: number
  total_revenue: number
  capacity_factor: number
  avg_wind_speed: number
  avg_price: number
  peak_power_kw: number
  hours_at_rated: number
  hours_below_cutin: number
  iso_region: string | null
  pricing_mode: string
  data_period_days: number
}

export interface IsoInfo {
  name: string
  mean_price: number
  min_price: number
  max_price: number
}

export interface TimeSeriesData {
  timestamps: string[]
  wind_speeds: number[]
  power_outputs: number[]
  energy_outputs: number[]
  prices: number[]
  revenues: number[]
}

export interface MonthlyStats {
  months: string[]
  energy_mwh: number[]
  revenue: number[]
  capacity_factors: number[]
  avg_wind_speed: number[]
  avg_price: number[]
}

export interface HourlyPatternData {
  hours: number[]
  avg_power_by_hour: number[]
  avg_wind_by_hour: number[]
  avg_price_by_hour: number[]
}

export interface WindDistributionData {
  bins: number[]
  frequencies: number[]
  power_contribution: number[]
}

export interface SeasonalData {
  seasons: string[]
  energy_mwh: number[]
  capacity_factors: number[]
  avg_wind_speed: number[]
}

export interface AnalyticsResponse {
  summary: SummaryStats
  timeseries: TimeSeriesData
  monthly: MonthlyStats
  hourly_pattern: HourlyPatternData
  wind_distribution: WindDistributionData
  seasonal: SeasonalData
}
