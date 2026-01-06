import type { 
  Project, 
  ProjectFormData, 
  WindDataResponse, 
  RevenueDataResponse,
  AnalyticsRequest,
  AnalyticsResponse,
  IsoInfo,
} from './types'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new ApiError(
      response.status,
      errorData.detail || `Request failed with status ${response.status}`
    )
  }

  return response.json()
}

// Wind data API
export const windApi = {
  getWindData: async (params: {
    latitude: number
    longitude: number
    start_date: string
    end_date: string
    hub_height_m?: number
    rotor_diameter_m?: number
    rated_power_kw?: number
  }): Promise<WindDataResponse> => {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, String(value))
      }
    })
    return fetchApi(`/api/wind?${searchParams}`)
  },

  getRevenueData: async (params: {
    latitude: number
    longitude: number
    start_date: string
    end_date: string
    hub_height_m?: number
    rotor_diameter_m?: number
    rated_power_kw?: number
  }): Promise<RevenueDataResponse> => {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, String(value))
      }
    })
    return fetchApi(`/api/revenue?${searchParams}`)
  },
}

// Projects API (calls FastAPI which proxies to Supabase)
export const projectsApi = {
  getAll: async (userId: string): Promise<Project[]> => {
    return fetchApi(`/api/projects?user_id=${userId}`)
  },

  getById: async (id: string): Promise<Project> => {
    return fetchApi(`/api/projects/${id}`)
  },

  create: async (userId: string, data: ProjectFormData): Promise<Project> => {
    return fetchApi('/api/projects', {
      method: 'POST',
      body: JSON.stringify({ ...data, user_id: userId }),
    })
  },

  update: async (id: string, data: Partial<ProjectFormData>): Promise<Project> => {
    return fetchApi(`/api/projects/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    })
  },

  delete: async (id: string): Promise<void> => {
    return fetchApi(`/api/projects/${id}`, {
      method: 'DELETE',
    })
  },

  updateCachedResults: async (
    id: string,
    results: {
      cached_annual_revenue: number
      cached_capacity_factor: number
      cached_iso: string | null
    }
  ): Promise<Project> => {
    return fetchApi(`/api/projects/${id}/cache`, {
      method: 'POST',
      body: JSON.stringify(results),
    })
  },
}

// Prices API
export const pricesApi = {
  getForLocation: async (params: {
    latitude: number
    longitude: number
    start_date: string
    end_date: string
  }) => {
    const searchParams = new URLSearchParams()
    Object.entries(params).forEach(([key, value]) => {
      searchParams.append(key, String(value))
    })
    return fetchApi(`/api/prices?${searchParams}`)
  },

  getIsoList: async (): Promise<string[]> => {
    return fetchApi('/api/prices/isos')
  },
}

// Reports API
export interface ReportRequest {
  project_id: string
  project_name: string
  user_id: string
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
  save_to_account: boolean
}

export interface ReportResponse {
  report_id: string | null
  html_content: string
  generated_at: string
  saved: boolean
}

export interface SavedReport {
  id: string
  project_id: string
  report_type: string
  created_at: string
  parameters: Record<string, unknown>
  summary: {
    total_energy_mwh: number
    total_revenue: number
    capacity_factor: number
    avg_price: number
  }
}

export const reportsApi = {
  generate: async (params: ReportRequest): Promise<ReportResponse> => {
    return fetchApi('/api/reports', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  getUserReports: async (userId: string): Promise<SavedReport[]> => {
    return fetchApi(`/api/reports/user/${userId}`)
  },

  getReportHtml: async (reportId: string): Promise<string> => {
    const response = await fetch(`${API_BASE_URL}/api/reports/html/${reportId}`)
    if (!response.ok) {
      throw new Error('Failed to fetch report')
    }
    return response.text()
  },

  deleteReport: async (reportId: string, userId: string): Promise<void> => {
    return fetchApi(`/api/reports/${reportId}?user_id=${userId}`, {
      method: 'DELETE',
    })
  },
}

// Analytics API
export const analyticsApi = {
  getProjectAnalytics: async (params: AnalyticsRequest): Promise<AnalyticsResponse> => {
    return fetchApi('/api/analytics', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  },

  getIsoPrices: async (): Promise<Record<string, number>> => {
    return fetchApi('/api/analytics/iso-prices')
  },

  getAvailableIsos: async (): Promise<IsoInfo[]> => {
    return fetchApi('/api/analytics/isos')
  },
}

// Health check
export const healthCheck = async (): Promise<{ status: string }> => {
  return fetchApi('/api/health')
}
