'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import dynamic from 'next/dynamic'
import { useAuth } from '@/hooks/useAuth'
import { useProjects, useCreateProject, useUpdateProject } from '@/hooks/useProjects'
import { ProjectModal } from '@/components/Projects/ProjectModal'
import { HeatmapLegend } from '@/components/Map/HeatmapLegend'
import type { Project, ProjectFormData } from '@/lib/types'

// Dynamically import map component to avoid SSR issues with Leaflet
const MapContainer = dynamic(
  () => import('@/components/Map/MapContainer'),
  { 
    ssr: false,
    loading: () => (
      <div className="w-full h-[600px] bg-gray-100 animate-pulse rounded-lg flex items-center justify-center">
        <span className="text-gray-500">Loading map...</span>
      </div>
    )
  }
)

export default function DashboardPage() {
  const router = useRouter()
  const { user } = useAuth()
  const { data: projects, isLoading } = useProjects(user?.id)
  const createProject = useCreateProject()
  const updateProject = useUpdateProject()
  
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)
  const [prospectLocation, setProspectLocation] = useState<{lat: number, lng: number} | null>(null)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [showHeatmap, setShowHeatmap] = useState(true)

  const handleMapClick = (lat: number, lng: number) => {
    setProspectLocation({ lat, lng })
    setSelectedProject(null)
  }

  const handleMarkerClick = (project: Project) => {
    setSelectedProject(project)
    setProspectLocation(null)
  }

  const handleMarkerDragEnd = async (projectId: string, lat: number, lng: number) => {
    try {
      await updateProject.mutateAsync({
        id: projectId,
        data: { latitude: lat, longitude: lng }
      })
    } catch (error) {
      console.error('Failed to update project location:', error)
    }
  }

  const handleOpenModal = () => {
    if (!user) {
      // Redirect to login if not authenticated
      window.location.href = '/login'
      return
    }
    setIsModalOpen(true)
  }

  const handleSaveProject = async (data: ProjectFormData) => {
    if (!user) return
    
    await createProject.mutateAsync({
      userId: user.id,
      data
    })
    
    setIsModalOpen(false)
    setProspectLocation(null)
  }

  const handleSelectExisting = (project: Project) => {
    setSelectedProject(project)
    setProspectLocation({ lat: project.latitude, lng: project.longitude })
  }

  const navigateToAnalytics = (project: Project) => {
    // Build URL params for analytics page
    const params = new URLSearchParams({
      projectId: project.id,
      name: project.name,
      lat: project.latitude.toString(),
      lng: project.longitude.toString(),
      hub: project.hub_height_m.toString(),
      rotor: project.rotor_diameter_m.toString(),
      power: (project.rated_power_kw || 3000).toString(),
    })
    
    // Default to last year of data
    const endDate = new Date()
    endDate.setDate(endDate.getDate() - 6) // 5-day lag for archive data
    const startDate = new Date(endDate)
    startDate.setFullYear(startDate.getFullYear() - 1)
    
    params.set('start', startDate.toISOString().split('T')[0])
    params.set('end', endDate.toISOString().split('T')[0])
    
    router.push(`/analytics?${params.toString()}`)
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <div className="flex items-center gap-4">
          {/* Heatmap toggle */}
          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={(e) => setShowHeatmap(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            Show Revenue Heatmap
          </label>
          {user && (
            <span className="text-sm text-gray-500">
              {projects?.length ?? 0} project{projects?.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Map Section */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <MapContainer
              projects={projects ?? []}
              prospectLocation={prospectLocation}
              onMapClick={handleMapClick}
              onMarkerClick={handleMarkerClick}
              onMarkerDragEnd={handleMarkerDragEnd}
              selectedProjectId={selectedProject?.id}
              showHeatmap={showHeatmap}
            />
          </div>
          {/* Heatmap legend */}
          <HeatmapLegend visible={showHeatmap} />
        </div>

        {/* Side Panel */}
        <div className="space-y-4">
          {/* Quick Stats */}
          {user && projects && projects.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Summary</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Projects</span>
                  <span className="font-medium">{projects.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Est. Annual Revenue</span>
                  <span className="font-medium">
                    ${projects.reduce((sum, p) => sum + (p.cached_annual_revenue ?? 0), 0).toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg Capacity Factor</span>
                  <span className="font-medium">
                    {projects.filter(p => p.cached_capacity_factor).length > 0
                      ? (projects.reduce((sum, p) => sum + (p.cached_capacity_factor ?? 0), 0) / projects.filter(p => p.cached_capacity_factor).length * 100).toFixed(1)
                      : '—'}%
                  </span>
                </div>
              </div>
              {/* Project selector for analysis */}
              <div className="mt-4 pt-4 border-t border-gray-100">
                <label className="block text-sm text-gray-600 mb-2">Analyze Project</label>
                <select 
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-blue-500 focus:border-blue-500"
                  value={selectedProject?.id || ''}
                  onChange={(e) => {
                    const project = projects.find(p => p.id === e.target.value)
                    if (project) {
                      setSelectedProject(project)
                    }
                  }}
                >
                  <option value="">Select a project...</option>
                  {projects.map(p => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
                <button
                  onClick={() => selectedProject && navigateToAnalytics(selectedProject)}
                  disabled={!selectedProject}
                  className={`w-full mt-2 px-4 py-2 text-sm rounded-lg transition-colors ${
                    selectedProject 
                      ? 'bg-blue-600 text-white hover:bg-blue-700' 
                      : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  View Analytics
                </button>
              </div>
            </div>
          )}

          {/* Prospect Location Panel */}
          {prospectLocation && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">New Location</h2>
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-500">Coordinates</span>
                  <p className="font-mono text-sm">
                    {prospectLocation.lat.toFixed(4)}°N, {Math.abs(prospectLocation.lng).toFixed(4)}°W
                  </p>
                </div>
                <button
                  onClick={handleOpenModal}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Create Project Here
                </button>
                <button
                  onClick={() => setProspectLocation(null)}
                  className="w-full px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Clear Selection
                </button>
              </div>
            </div>
          )}

          {/* Selected Project Panel */}
          {selectedProject && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">{selectedProject.name}</h2>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Location</span>
                  <span className="font-mono">
                    {selectedProject.latitude.toFixed(4)}°N, {Math.abs(selectedProject.longitude).toFixed(4)}°W
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Hub Height</span>
                  <span>{selectedProject.hub_height_m}m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Rotor Diameter</span>
                  <span>{selectedProject.rotor_diameter_m}m</span>
                </div>
                {selectedProject.cached_iso && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">ISO Region</span>
                    <span>{selectedProject.cached_iso}</span>
                  </div>
                )}
                {selectedProject.cached_capacity_factor && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">Capacity Factor</span>
                    <span>{(selectedProject.cached_capacity_factor * 100).toFixed(1)}%</span>
                  </div>
                )}
                {selectedProject.cached_annual_revenue && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">Est. Annual Revenue</span>
                    <span>${selectedProject.cached_annual_revenue.toLocaleString()}</span>
                  </div>
                )}
                <div className="pt-3 border-t border-gray-100 flex gap-2">
                  <button 
                    onClick={() => navigateToAnalytics(selectedProject)}
                    className="flex-1 px-3 py-2 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
                  >
                    Analyze
                  </button>
                  <button className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded hover:bg-gray-50">
                    Generate Report
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Getting Started */}
          {!user && (
            <div className="bg-blue-50 rounded-lg border border-blue-200 p-4">
              <h2 className="text-lg font-semibold text-blue-900 mb-2">Getting Started</h2>
              <p className="text-sm text-blue-700 mb-4">
                Sign in to save projects and access detailed wind analysis.
              </p>
              <a
                href="/login"
                className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
              >
                Sign In
              </a>
            </div>
          )}

          {user && (!projects || projects.length === 0) && !isLoading && !prospectLocation && (
            <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
              <h2 className="text-lg font-semibold text-gray-900 mb-2">No Projects Yet</h2>
              <p className="text-sm text-gray-600 mb-4">
                Click anywhere on the map to start analyzing a new location.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Project Modal */}
      <ProjectModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveProject}
        onSelectExisting={handleSelectExisting}
        existingProjects={projects ?? []}
        initialLocation={prospectLocation}
        isLoading={createProject.isPending}
      />
    </div>
  )
}
