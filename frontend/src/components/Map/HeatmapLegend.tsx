'use client'

import { useEffect, useState } from 'react'

interface LegendData {
  min: number
  max: number
  median: number
  mean: number
  unit: string
  description: string
}

interface HeatmapLegendProps {
  visible: boolean
}

function formatCurrency(value: number): string {
  if (value >= 1000000) {
    return `$${(value / 1000000).toFixed(1)}M`
  } else if (value >= 1000) {
    return `$${(value / 1000).toFixed(0)}K`
  }
  return `$${value.toFixed(0)}`
}

export function HeatmapLegend({ visible }: HeatmapLegendProps) {
  const [legendData, setLegendData] = useState<LegendData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!visible) return

    const fetchLegend = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        const response = await fetch(`${apiUrl}/api/heatmap/legend`)
        if (response.ok) {
          const data = await response.json()
          setLegendData(data)
        }
      } catch (error) {
        console.error('Failed to fetch heatmap legend:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchLegend()
  }, [visible])

  if (!visible) return null

  return (
    <div className="mt-3 p-3 bg-white rounded-lg border border-gray-200 shadow-sm">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700">
          Annual Revenue Potential
        </span>
        {legendData && (
          <span className="text-xs text-gray-500">
            (3MW Turbine, 2024 prices)
          </span>
        )}
      </div>
      
      {/* Color Spectrum */}
      <div className="relative">
        {/* Gradient bar */}
        <div 
          className="h-4 rounded-sm"
          style={{
            background: 'linear-gradient(to right, rgb(0, 200, 0), rgb(255, 255, 0), rgb(255, 0, 0))'
          }}
        />
        
        {/* Tick marks and labels */}
        {legendData && !loading ? (
          <div className="relative mt-1">
            {/* Min value */}
            <div className="absolute left-0 text-xs text-gray-600">
              <div className="w-px h-2 bg-gray-400 mb-0.5" />
              <span>{formatCurrency(legendData.min)}</span>
            </div>
            
            {/* Median marker (positioned proportionally) */}
            {(() => {
              const medianPosition = ((legendData.median - legendData.min) / (legendData.max - legendData.min)) * 100
              return (
                <div 
                  className="absolute text-xs text-gray-600 transform -translate-x-1/2"
                  style={{ left: `${Math.min(Math.max(medianPosition, 10), 90)}%` }}
                >
                  <div className="w-px h-2 bg-gray-400 mb-0.5 mx-auto" />
                  <span className="whitespace-nowrap">
                    {formatCurrency(legendData.median)}
                    <span className="text-gray-400 ml-1">(median)</span>
                  </span>
                </div>
              )
            })()}
            
            {/* Max value */}
            <div className="absolute right-0 text-xs text-gray-600 text-right">
              <div className="w-px h-2 bg-gray-400 mb-0.5 ml-auto" />
              <span>{formatCurrency(legendData.max)}</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-between mt-1 text-xs text-gray-400">
            <span>Low</span>
            <span>High</span>
          </div>
        )}
      </div>
      
      {/* Additional stats */}
      {legendData && !loading && (
        <div className="mt-4 pt-2 border-t border-gray-100 grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-gray-500">Mean: </span>
            <span className="font-medium text-gray-700">{formatCurrency(legendData.mean)}/yr</span>
          </div>
          <div>
            <span className="text-gray-500">Median: </span>
            <span className="font-medium text-gray-700">{formatCurrency(legendData.median)}/yr</span>
          </div>
        </div>
      )}
    </div>
  )
}
