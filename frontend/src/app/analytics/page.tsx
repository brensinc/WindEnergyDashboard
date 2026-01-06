'use client'

import { useEffect, useState, Suspense, useCallback } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import { analyticsApi, reportsApi } from '@/lib/api'
import { useAuth } from '@/hooks/useAuth'
import type { AnalyticsResponse, IsoInfo } from '@/lib/types'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as React.ComponentType<{
  data: Plotly.Data[]
  layout: Partial<Plotly.Layout>
  config?: Partial<Plotly.Config>
  style?: React.CSSProperties
}>

// Type for Plotly - needed because react-plotly.js types are incomplete
declare namespace Plotly {
  interface Data {
    x?: (string | number)[]
    y?: (string | number)[]
    type?: string
    mode?: string
    name?: string
    marker?: Record<string, unknown>
    line?: Record<string, unknown>
    fill?: string
    fillcolor?: string
    yaxis?: string
    labels?: string[]
    values?: number[]
    textinfo?: string
    hovertemplate?: string
  }
  interface Layout {
    height?: number
    margin?: { t?: number; r?: number; b?: number; l?: number }
    xaxis?: Record<string, unknown>
    yaxis?: Record<string, unknown>
    yaxis2?: Record<string, unknown>
    legend?: Record<string, unknown>
    barmode?: string
    bargap?: number
    showlegend?: boolean
    hovermode?: string | boolean
  }
  interface Config {
    displayModeBar?: boolean
    responsive?: boolean
  }
}

type TabId = 'overview' | 'timeseries' | 'monthly' | 'hourly' | 'distribution' | 'seasonal'

const tabs: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'timeseries', label: 'Time Series' },
  { id: 'monthly', label: 'Monthly' },
  { id: 'hourly', label: 'Hourly Patterns' },
  { id: 'distribution', label: 'Wind Distribution' },
  { id: 'seasonal', label: 'Seasonal' },
]

function AnalyticsContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const { user } = useAuth()
  
  const [activeTab, setActiveTab] = useState<TabId>('overview')
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [availableIsos, setAvailableIsos] = useState<IsoInfo[]>([])
  
  // Pricing options state
  const [pricingMode, setPricingMode] = useState<'market' | 'fixed'>('market')
  const [fixedPrice, setFixedPrice] = useState(50)
  const [selectedIso, setSelectedIso] = useState<string>('')
  
  // Report generation state
  const [generatingReport, setGeneratingReport] = useState(false)
  const [reportGenerated, setReportGenerated] = useState(false)

  // Get project info from URL params
  const projectId = searchParams.get('projectId') || ''
  const projectName = searchParams.get('name') || 'Project'
  const latitude = parseFloat(searchParams.get('lat') || '0')
  const longitude = parseFloat(searchParams.get('lng') || '0')
  const hubHeight = parseInt(searchParams.get('hub') || '100')
  const rotorDiameter = parseInt(searchParams.get('rotor') || '100')
  const ratedPower = parseInt(searchParams.get('power') || '3000')
  const startDate = searchParams.get('start') || getDefaultStartDate()
  const endDate = searchParams.get('end') || getDefaultEndDate()

  function getDefaultStartDate(): string {
    const date = new Date()
    date.setFullYear(date.getFullYear() - 1)
    return date.toISOString().split('T')[0]
  }

  function getDefaultEndDate(): string {
    const date = new Date()
    date.setDate(date.getDate() - 6) // 5-day lag for archive data
    return date.toISOString().split('T')[0]
  }

  // Fetch available ISOs on mount
  useEffect(() => {
    analyticsApi.getAvailableIsos()
      .then(setAvailableIsos)
      .catch(console.error)
  }, [])

  const fetchAnalytics = useCallback(async () => {
    if (!projectId || !latitude || !longitude) {
      setError('Missing project parameters')
      setLoading(false)
      return
    }

    try {
      setLoading(true)
      setError(null)
      
      const requestParams = {
        project_id: projectId,
        latitude,
        longitude,
        hub_height_m: hubHeight,
        rotor_diameter_m: rotorDiameter,
        rated_power_kw: ratedPower,
        start_date: startDate,
        end_date: endDate,
        pricing_mode: pricingMode,
        fixed_price: fixedPrice,
        iso_override: selectedIso || undefined,
      }
      console.log('Sending analytics request:', requestParams)
      
      const data = await analyticsApi.getProjectAnalytics(requestParams)
      
      console.log('Response pricing_mode:', data.summary.pricing_mode, 'avg_price:', data.summary.avg_price)
      setAnalytics(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load analytics')
    } finally {
      setLoading(false)
    }
  }, [projectId, latitude, longitude, hubHeight, rotorDiameter, ratedPower, startDate, endDate, pricingMode, fixedPrice, selectedIso])

  // Initial fetch
  useEffect(() => {
    fetchAnalytics()
  }, []) // Only run once on mount

  // Refetch when pricing options change
  const handleRefresh = () => {
    fetchAnalytics()
  }

  // Generate report
  const handleGenerateReport = async (saveToAccount: boolean) => {
    if (!user && saveToAccount) {
      alert('Please sign in to save reports to your account')
      return
    }

    try {
      setGeneratingReport(true)
      
      const response = await reportsApi.generate({
        project_id: projectId,
        project_name: projectName,
        user_id: user?.id || '',
        latitude,
        longitude,
        hub_height_m: hubHeight,
        rotor_diameter_m: rotorDiameter,
        rated_power_kw: ratedPower,
        start_date: startDate,
        end_date: endDate,
        pricing_mode: pricingMode,
        fixed_price: fixedPrice,
        iso_override: selectedIso || undefined,
        save_to_account: saveToAccount,
      })

      // Open report in new tab or download
      const blob = new Blob([response.html_content], { type: 'text/html' })
      const url = URL.createObjectURL(blob)
      window.open(url, '_blank')
      
      if (saveToAccount && response.saved) {
        setReportGenerated(true)
        setTimeout(() => setReportGenerated(false), 3000)
      }
    } catch (err) {
      console.error('Failed to generate report:', err)
      alert('Failed to generate report. Please try again.')
    } finally {
      setGeneratingReport(false)
    }
  }

  if (loading && !analytics) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading analytics...</p>
          <p className="text-sm text-gray-400 mt-2">Fetching weather data and calculating metrics</p>
        </div>
      </div>
    )
  }

  if (error && !analytics) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-sm border border-red-200 max-w-md">
          <h2 className="text-lg font-semibold text-red-600 mb-2">Error Loading Analytics</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => router.back()}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
          >
            Go Back
          </button>
        </div>
      </div>
    )
  }

  if (!analytics) return null

  const { summary, timeseries, monthly, hourly_pattern, wind_distribution, seasonal } = analytics

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <button
                onClick={() => router.back()}
                className="text-sm text-blue-600 hover:text-blue-800 mb-1 flex items-center gap-1"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Dashboard
              </button>
              <h1 className="text-2xl font-bold text-gray-900">
                {projectName} Analytics
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                {latitude.toFixed(4)}°N, {Math.abs(longitude).toFixed(4)}°W | {startDate} to {endDate}
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm text-gray-500">ISO Region</p>
                <p className="text-lg font-semibold text-gray-900">{summary.iso_region || 'N/A'}</p>
              </div>
              
              {/* Generate Report Dropdown */}
              <div className="relative">
                <button
                  onClick={() => handleGenerateReport(false)}
                  disabled={generatingReport}
                  className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
                    generatingReport
                      ? 'bg-gray-100 text-gray-400'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {generatingReport ? (
                    <>
                      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Generating...
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Generate Report
                    </>
                  )}
                </button>
                {user && (
                  <button
                    onClick={() => handleGenerateReport(true)}
                    disabled={generatingReport}
                    className="mt-1 w-full text-xs text-blue-600 hover:text-blue-800"
                  >
                    Generate & Save to Account
                  </button>
                )}
                {reportGenerated && (
                  <div className="absolute top-full right-0 mt-2 px-3 py-1 bg-green-100 text-green-700 text-sm rounded shadow">
                    Report saved!
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Pricing Controls */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-gray-700">Pricing:</label>
              <select
                value={pricingMode}
                onChange={(e) => setPricingMode(e.target.value as 'market' | 'fixed')}
                className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="market">Market (Hourly)</option>
                <option value="fixed">Fixed Price</option>
              </select>
            </div>

            {pricingMode === 'fixed' ? (
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">Price:</label>
                <div className="flex items-center">
                  <span className="text-sm text-gray-500">$</span>
                  <input
                    type="number"
                    value={fixedPrice}
                    onChange={(e) => setFixedPrice(parseFloat(e.target.value) || 0)}
                    className="w-20 text-sm border border-gray-300 rounded px-2 py-1 ml-1"
                    min={0}
                    max={500}
                  />
                  <span className="text-sm text-gray-500 ml-1">/MWh</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-600">ISO:</label>
                <select
                  value={selectedIso}
                  onChange={(e) => setSelectedIso(e.target.value)}
                  className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Auto-detect</option>
                  {availableIsos.map((iso) => (
                    <option key={iso.name} value={iso.name}>
                      {iso.name} (avg ${iso.mean_price.toFixed(0)}/MWh)
                    </option>
                  ))}
                </select>
              </div>
            )}

            <button
              onClick={handleRefresh}
              disabled={loading}
              className={`px-3 py-1 text-sm rounded ${
                loading
                  ? 'bg-gray-100 text-gray-400'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {loading ? 'Loading...' : 'Refresh'}
            </button>

            <div className="text-sm text-gray-500 ml-auto">
              Mode: <span className="font-medium">{summary.pricing_mode}</span> | 
              Avg: <span className="font-medium">${summary.avg_price.toFixed(2)}/MWh</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex gap-1 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium whitespace-nowrap border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'overview' && (
          <OverviewTab summary={summary} monthly={monthly} seasonal={seasonal} />
        )}
        {activeTab === 'timeseries' && (
          <TimeSeriesTab data={timeseries} />
        )}
        {activeTab === 'monthly' && (
          <MonthlyTab data={monthly} />
        )}
        {activeTab === 'hourly' && (
          <HourlyTab data={hourly_pattern} />
        )}
        {activeTab === 'distribution' && (
          <DistributionTab data={wind_distribution} />
        )}
        {activeTab === 'seasonal' && (
          <SeasonalTab data={seasonal} />
        )}
      </div>
    </div>
  )
}

// Overview Tab
function OverviewTab({ 
  summary, 
  monthly, 
  seasonal 
}: { 
  summary: AnalyticsResponse['summary']
  monthly: AnalyticsResponse['monthly']
  seasonal: AnalyticsResponse['seasonal']
}) {
  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard 
          label="Total Energy" 
          value={`${(summary.total_energy_mwh / 1000).toFixed(1)} GWh`}
          subtext={`${summary.data_period_days} days`}
        />
        <KpiCard 
          label="Total Revenue" 
          value={`$${(summary.total_revenue / 1000).toFixed(0)}K`}
          subtext={`Avg $${summary.avg_price.toFixed(2)}/MWh`}
        />
        <KpiCard 
          label="Capacity Factor" 
          value={`${(summary.capacity_factor * 100).toFixed(1)}%`}
          subtext={`${summary.hours_at_rated} hrs at rated`}
        />
        <KpiCard 
          label="Avg Wind Speed" 
          value={`${summary.avg_wind_speed.toFixed(1)} m/s`}
          subtext={`${summary.hours_below_cutin} hrs below cut-in`}
        />
      </div>

      {/* Mini Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Monthly Revenue Trend */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-4">Monthly Revenue</h3>
          <Plot
            data={[
              {
                x: monthly.months,
                y: monthly.revenue,
                type: 'bar',
                marker: { color: '#3b82f6' },
              },
            ]}
            layout={{
              height: 250,
              margin: { t: 10, r: 20, b: 40, l: 60 },
              xaxis: { tickangle: -45 },
              yaxis: { title: 'Revenue ($)' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        {/* Seasonal Capacity Factor */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-4">Seasonal Performance</h3>
          <Plot
            data={[
              {
                x: seasonal.seasons,
                y: seasonal.capacity_factors.map(cf => cf * 100),
                type: 'bar',
                marker: { 
                  color: ['#60a5fa', '#34d399', '#fbbf24', '#f97316'],
                },
              },
            ]}
            layout={{
              height: 250,
              margin: { t: 10, r: 20, b: 40, l: 60 },
              yaxis: { title: 'Capacity Factor (%)' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
    </div>
  )
}

function KpiCard({ label, value, subtext }: { label: string; value: string; subtext: string }) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="text-2xl font-bold text-gray-900 mt-1">{value}</p>
      <p className="text-xs text-gray-400 mt-1">{subtext}</p>
    </div>
  )
}

// Time Series Tab
function TimeSeriesTab({ data }: { data: AnalyticsResponse['timeseries'] }) {
  const [metric, setMetric] = useState<'power' | 'wind' | 'revenue' | 'price'>('power')

  // Debug: log the data to verify revenues are different from power_outputs
  console.log('TimeSeriesTab data:', {
    power_outputs_sample: data.power_outputs?.slice(0, 3),
    revenues_sample: data.revenues?.slice(0, 3),
  })

  const getChartData = () => {
    switch (metric) {
      case 'power':
        return { y: data.power_outputs, title: 'Power Output (kW)', color: '#3b82f6' }
      case 'wind':
        return { y: data.wind_speeds, title: 'Wind Speed (m/s)', color: '#10b981' }
      case 'revenue':
        return { y: data.revenues, title: 'Revenue ($)', color: '#f59e0b' }
      case 'price':
        return { y: data.prices, title: 'Price ($/MWh)', color: '#8b5cf6' }
    }
  }

  const chartData = getChartData()
  console.log('Selected metric:', metric, 'chartData.y sample:', chartData.y?.slice(0, 3))

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-gray-900">Time Series Data</h3>
        <div className="flex gap-2">
          {(['power', 'wind', 'revenue', 'price'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`px-3 py-1 text-sm rounded ${
                metric === m
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {m.charAt(0).toUpperCase() + m.slice(1)}
            </button>
          ))}
        </div>
      </div>
      <Plot
        key={metric}
        data={[
          {
            x: data.timestamps,
            y: chartData.y,
            type: 'scatter',
            mode: 'lines',
            line: { color: chartData.color, width: 1 },
            fill: 'tozeroy',
            fillcolor: `${chartData.color}20`,
          },
        ]}
        layout={{
          height: 400,
          margin: { t: 20, r: 20, b: 50, l: 60 },
          xaxis: { title: 'Date' },
          yaxis: { title: chartData.title },
          hovermode: 'x unified',
        }}
        config={{ responsive: true }}
        style={{ width: '100%' }}
      />
    </div>
  )
}

// Monthly Tab
function MonthlyTab({ data }: { data: AnalyticsResponse['monthly'] }) {
  return (
    <div className="space-y-6">
      {/* Energy & Revenue */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Monthly Energy & Revenue</h3>
        <Plot
          data={[
            {
              x: data.months,
              y: data.energy_mwh,
              type: 'bar',
              name: 'Energy (MWh)',
              marker: { color: '#3b82f6' },
              yaxis: 'y',
            },
            {
              x: data.months,
              y: data.revenue,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Revenue ($)',
              line: { color: '#10b981', width: 2 },
              marker: { size: 6 },
              yaxis: 'y2',
            },
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 60, b: 50, l: 60 },
            xaxis: { title: 'Month', tickangle: -45 },
            yaxis: { title: 'Energy (MWh)', side: 'left' },
            yaxis2: { 
              title: 'Revenue ($)', 
              side: 'right', 
              overlaying: 'y',
            },
            legend: { orientation: 'h', y: 1.1 },
            barmode: 'group',
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      {/* Price & Capacity Factor */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Monthly Price & Performance</h3>
        <Plot
          data={[
            {
              x: data.months,
              y: data.avg_price,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Avg Price ($/MWh)',
              line: { color: '#8b5cf6', width: 2 },
              marker: { size: 6 },
            },
            {
              x: data.months,
              y: data.capacity_factors.map(cf => cf * 100),
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Capacity Factor (%)',
              line: { color: '#f59e0b', width: 2 },
              marker: { size: 6 },
              yaxis: 'y2',
            },
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 60, b: 50, l: 60 },
            xaxis: { title: 'Month', tickangle: -45 },
            yaxis: { title: 'Price ($/MWh)', side: 'left' },
            yaxis2: { 
              title: 'Capacity Factor (%)', 
              side: 'right', 
              overlaying: 'y',
            },
            legend: { orientation: 'h', y: 1.1 },
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>
    </div>
  )
}

// Hourly Pattern Tab
function HourlyTab({ data }: { data: AnalyticsResponse['hourly_pattern'] }) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Average Daily Pattern</h3>
        <Plot
          data={[
            {
              x: data.hours,
              y: data.avg_power_by_hour,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Avg Power (kW)',
              line: { color: '#3b82f6', width: 2 },
              fill: 'tozeroy',
              fillcolor: 'rgba(59, 130, 246, 0.1)',
            },
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 20, b: 50, l: 60 },
            xaxis: { 
              title: 'Hour of Day',
              tickmode: 'array',
              tickvals: [0, 3, 6, 9, 12, 15, 18, 21],
              ticktext: ['12am', '3am', '6am', '9am', '12pm', '3pm', '6pm', '9pm'],
            },
            yaxis: { title: 'Average Power (kW)' },
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Hourly Price Pattern</h3>
        <Plot
          data={[
            {
              x: data.hours,
              y: data.avg_price_by_hour,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Avg Price ($/MWh)',
              line: { color: '#8b5cf6', width: 2 },
              fill: 'tozeroy',
              fillcolor: 'rgba(139, 92, 246, 0.1)',
            },
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 20, b: 50, l: 60 },
            xaxis: { 
              title: 'Hour of Day',
              tickmode: 'array',
              tickvals: [0, 3, 6, 9, 12, 15, 18, 21],
              ticktext: ['12am', '3am', '6am', '9am', '12pm', '3pm', '6pm', '9pm'],
            },
            yaxis: { title: 'Average Price ($/MWh)' },
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
        <p className="text-sm text-gray-500 mt-2">
          Shows how electricity prices vary throughout the day. 
          Peak prices typically occur during afternoon/evening hours (2pm-8pm).
        </p>
      </div>
    </div>
  )
}

// Distribution Tab
function DistributionTab({ data }: { data: AnalyticsResponse['wind_distribution'] }) {
  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Wind Speed Distribution</h3>
        <Plot
          data={[
            {
              x: data.bins,
              y: data.frequencies,
              type: 'bar',
              name: 'Frequency (%)',
              marker: { color: '#3b82f6' },
            },
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 20, b: 50, l: 60 },
            xaxis: { title: 'Wind Speed (m/s)' },
            yaxis: { title: 'Frequency (%)' },
            bargap: 0.1,
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Energy Contribution by Wind Speed</h3>
        <Plot
          data={[
            {
              x: data.bins,
              y: data.power_contribution,
              type: 'bar',
              name: 'Energy Contribution (%)',
              marker: { color: '#10b981' },
            },
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 20, b: 50, l: 60 },
            xaxis: { title: 'Wind Speed (m/s)' },
            yaxis: { title: 'Energy Contribution (%)' },
            bargap: 0.1,
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
        <p className="text-sm text-gray-500 mt-2">
          Shows what percentage of total energy comes from each wind speed range.
          Higher values at mid-range speeds (8-14 m/s) indicate optimal turbine operation.
        </p>
      </div>
    </div>
  )
}

// Seasonal Tab
function SeasonalTab({ data }: { data: AnalyticsResponse['seasonal'] }) {
  const seasonColors = {
    Winter: '#60a5fa',
    Spring: '#34d399',
    Summer: '#fbbf24',
    Fall: '#f97316',
  }

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Seasonal Energy Production</h3>
        <Plot
          data={[
            {
              labels: data.seasons,
              values: data.energy_mwh,
              type: 'pie',
              marker: { 
                colors: data.seasons.map(s => seasonColors[s as keyof typeof seasonColors] || '#888'),
              },
              textinfo: 'label+percent',
              hovertemplate: '%{label}<br>%{value:.0f} MWh<br>%{percent}<extra></extra>',
            },
          ]}
          layout={{
            height: 350,
            margin: { t: 20, r: 20, b: 20, l: 20 },
            showlegend: false,
          }}
          config={{ responsive: true }}
          style={{ width: '100%' }}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Capacity Factor by Season</h3>
          <Plot
            data={[
              {
                x: data.seasons,
                y: data.capacity_factors.map(cf => cf * 100),
                type: 'bar',
                marker: { 
                  color: data.seasons.map(s => seasonColors[s as keyof typeof seasonColors] || '#888'),
                },
              },
            ]}
            layout={{
              height: 280,
              margin: { t: 20, r: 20, b: 40, l: 60 },
              yaxis: { title: 'Capacity Factor (%)' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Avg Wind Speed by Season</h3>
          <Plot
            data={[
              {
                x: data.seasons,
                y: data.avg_wind_speed,
                type: 'bar',
                marker: { 
                  color: data.seasons.map(s => seasonColors[s as keyof typeof seasonColors] || '#888'),
                },
              },
            ]}
            layout={{
              height: 280,
              margin: { t: 20, r: 20, b: 40, l: 60 },
              yaxis: { title: 'Wind Speed (m/s)' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </div>
      </div>
    </div>
  )
}

export default function AnalyticsPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    }>
      <AnalyticsContent />
    </Suspense>
  )
}
