'use client'

import { useEffect, useRef, useState } from 'react'
import { MapContainer as LeafletMap, TileLayer, Marker, Popup, useMapEvents, ImageOverlay } from 'react-leaflet'
import L from 'leaflet'
import type { Project } from '@/lib/types'
import 'leaflet/dist/leaflet.css'

// Fix for default marker icons in Next.js
const defaultIcon = L.icon({
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
})

const prospectIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
})

const selectedIcon = L.icon({
  iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41]
})

L.Marker.prototype.options.icon = defaultIcon

interface MapContainerProps {
  projects: Project[]
  prospectLocation: { lat: number; lng: number } | null
  onMapClick: (lat: number, lng: number) => void
  onMarkerClick: (project: Project) => void
  onMarkerDragEnd?: (projectId: string, lat: number, lng: number) => void
  selectedProjectId?: string
  showHeatmap?: boolean
}

// Heatmap bounds from the cached data
const HEATMAP_BOUNDS: L.LatLngBoundsExpression = [
  [24.5, -125.0],  // Southwest corner [lat, lng]
  [49.5, -66.0]    // Northeast corner [lat, lng]
]

// Component to handle map events
function MapEventHandler({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) {
  useMapEvents({
    click: (e) => {
      onMapClick(e.latlng.lat, e.latlng.lng)
    },
  })
  return null
}

// Component to handle draggable prospect marker
function DraggableProspectMarker({ 
  position, 
  onDragEnd 
}: { 
  position: { lat: number; lng: number }
  onDragEnd: (lat: number, lng: number) => void 
}) {
  const markerRef = useRef<L.Marker>(null)

  const eventHandlers = {
    dragend() {
      const marker = markerRef.current
      if (marker != null) {
        const pos = marker.getLatLng()
        onDragEnd(pos.lat, pos.lng)
      }
    },
  }

  return (
    <Marker
      draggable={true}
      eventHandlers={eventHandlers}
      position={[position.lat, position.lng]}
      ref={markerRef}
      icon={prospectIcon}
    >
      <Popup>
        <div className="text-sm">
          <p className="font-semibold">New Location</p>
          <p className="font-mono text-xs">
            {position.lat.toFixed(4)}°N, {Math.abs(position.lng).toFixed(4)}°W
          </p>
          <p className="text-gray-500 mt-1">Drag to adjust position</p>
        </div>
      </Popup>
    </Marker>
  )
}

// Component to handle draggable project marker
function DraggableProjectMarker({ 
  project,
  isSelected,
  onClick,
  onDragEnd 
}: { 
  project: Project
  isSelected: boolean
  onClick: () => void
  onDragEnd: (lat: number, lng: number) => void 
}) {
  const markerRef = useRef<L.Marker>(null)

  const eventHandlers = {
    click: onClick,
    dragend() {
      const marker = markerRef.current
      if (marker != null) {
        const pos = marker.getLatLng()
        onDragEnd(pos.lat, pos.lng)
      }
    },
  }

  return (
    <Marker
      draggable={true}
      eventHandlers={eventHandlers}
      position={[project.latitude, project.longitude]}
      ref={markerRef}
      icon={isSelected ? selectedIcon : defaultIcon}
    >
      <Popup>
        <div className="text-sm">
          <p className="font-semibold">{project.name}</p>
          <p className="font-mono text-xs">
            {project.latitude.toFixed(4)}°N, {Math.abs(project.longitude).toFixed(4)}°W
          </p>
          {project.cached_capacity_factor && (
            <p className="text-gray-600">
              CF: {(project.cached_capacity_factor * 100).toFixed(1)}%
            </p>
          )}
        </div>
      </Popup>
    </Marker>
  )
}

export default function MapContainer({
  projects,
  prospectLocation,
  onMapClick,
  onMarkerClick,
  onMarkerDragEnd,
  selectedProjectId,
  showHeatmap = true,
}: MapContainerProps) {
  const [prospectPos, setProspectPos] = useState(prospectLocation)
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null)
  const [heatmapError, setHeatmapError] = useState(false)

  // Load heatmap image URL
  useEffect(() => {
    if (showHeatmap) {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      setHeatmapUrl(`${apiUrl}/api/heatmap/image.png`)
    }
  }, [showHeatmap])

  // Update local state when prop changes
  useEffect(() => {
    setProspectPos(prospectLocation)
  }, [prospectLocation])

  const handleProspectDragEnd = (lat: number, lng: number) => {
    setProspectPos({ lat, lng })
    onMapClick(lat, lng) // Update parent state
  }

  const handleProjectDragEnd = (projectId: string, lat: number, lng: number) => {
    if (onMarkerDragEnd) {
      onMarkerDragEnd(projectId, lat, lng)
    }
  }

  // Center of the continental US
  const defaultCenter: [number, number] = [39.8283, -98.5795]
  const defaultZoom = 4

  return (
    <LeafletMap
      center={defaultCenter}
      zoom={defaultZoom}
      className="w-full h-[600px]"
      scrollWheelZoom={true}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      
      {/* Heatmap overlay */}
      {showHeatmap && heatmapUrl && !heatmapError && (
        <ImageOverlay
          url={heatmapUrl}
          bounds={HEATMAP_BOUNDS}
          opacity={0.6}
          eventHandlers={{
            error: () => setHeatmapError(true)
          }}
        />
      )}
      
      <MapEventHandler onMapClick={onMapClick} />
      
      {/* Project markers */}
      {projects.map((project) => (
        <DraggableProjectMarker
          key={project.id}
          project={project}
          isSelected={project.id === selectedProjectId}
          onClick={() => onMarkerClick(project)}
          onDragEnd={(lat, lng) => handleProjectDragEnd(project.id, lat, lng)}
        />
      ))}
      
      {/* Prospect marker (for new location) */}
      {prospectPos && (
        <DraggableProspectMarker
          position={prospectPos}
          onDragEnd={handleProspectDragEnd}
        />
      )}
    </LeafletMap>
  )
}
