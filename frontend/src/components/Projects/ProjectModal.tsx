'use client'

import { useState } from 'react'
import type { Project, ProjectFormData } from '@/lib/types'

interface ProjectModalProps {
  isOpen: boolean
  onClose: () => void
  onSave: (data: ProjectFormData) => Promise<void>
  onSelectExisting: (project: Project) => void
  existingProjects: Project[]
  initialLocation: { lat: number; lng: number } | null
  isLoading?: boolean
}

export function ProjectModal({
  isOpen,
  onClose,
  onSave,
  onSelectExisting,
  existingProjects,
  initialLocation,
  isLoading = false,
}: ProjectModalProps) {
  const [mode, setMode] = useState<'new' | 'existing'>('new')
  const [formData, setFormData] = useState<ProjectFormData>({
    name: '',
    description: '',
    latitude: initialLocation?.lat ?? 0,
    longitude: initialLocation?.lng ?? 0,
    hub_height_m: 100,
    rotor_diameter_m: 100,
    rated_power_kw: 3000,
  })
  const [error, setError] = useState<string | null>(null)

  // Update form when location changes
  if (initialLocation && (formData.latitude !== initialLocation.lat || formData.longitude !== initialLocation.lng)) {
    setFormData(prev => ({
      ...prev,
      latitude: initialLocation.lat,
      longitude: initialLocation.lng,
    }))
  }

  if (!isOpen) return null

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!formData.name.trim()) {
      setError('Project name is required')
      return
    }

    try {
      await onSave(formData)
      // Reset form
      setFormData({
        name: '',
        description: '',
        latitude: 0,
        longitude: 0,
        hub_height_m: 100,
        rotor_diameter_m: 100,
        rated_power_kw: 3000,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save project')
    }
  }

  const handleSelectProject = (project: Project) => {
    onSelectExisting(project)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-[1000] flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50" 
        onClick={onClose}
      />
      
      {/* Modal */}
      <div className="relative bg-white rounded-lg shadow-xl w-full max-w-lg mx-4 max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900">
            {mode === 'new' ? 'Create New Project' : 'Select Existing Project'}
          </h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Mode Toggle */}
        <div className="flex border-b">
          <button
            onClick={() => setMode('new')}
            className={`flex-1 py-3 text-sm font-medium ${
              mode === 'new'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            New Project
          </button>
          <button
            onClick={() => setMode('existing')}
            className={`flex-1 py-3 text-sm font-medium ${
              mode === 'existing'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Existing Projects ({existingProjects.length})
          </button>
        </div>

        {/* Content */}
        <div className="p-4 overflow-y-auto max-h-[60vh]">
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
              {error}
            </div>
          )}

          {mode === 'new' ? (
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Location (read-only) */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Latitude
                  </label>
                  <input
                    type="text"
                    value={formData.latitude.toFixed(4)}
                    readOnly
                    className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-500 font-mono text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Longitude
                  </label>
                  <input
                    type="text"
                    value={formData.longitude.toFixed(4)}
                    readOnly
                    className="w-full px-3 py-2 bg-gray-50 border border-gray-300 rounded-lg text-gray-500 font-mono text-sm"
                  />
                </div>
              </div>

              {/* Name */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Project Name *
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                  placeholder="My Wind Project"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Optional description..."
                  rows={2}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Turbine Parameters */}
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Hub Height (m)
                  </label>
                  <input
                    type="number"
                    value={formData.hub_height_m}
                    onChange={(e) => setFormData(prev => ({ ...prev, hub_height_m: parseInt(e.target.value) || 100 }))}
                    min={10}
                    max={200}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Rotor Dia. (m)
                  </label>
                  <input
                    type="number"
                    value={formData.rotor_diameter_m}
                    onChange={(e) => setFormData(prev => ({ ...prev, rotor_diameter_m: parseInt(e.target.value) || 100 }))}
                    min={20}
                    max={200}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Rated Power (kW)
                  </label>
                  <input
                    type="number"
                    value={formData.rated_power_kw}
                    onChange={(e) => setFormData(prev => ({ ...prev, rated_power_kw: parseInt(e.target.value) || 3000 }))}
                    min={100}
                    max={20000}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>

              {/* Submit */}
              <div className="pt-4">
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? 'Creating...' : 'Create Project'}
                </button>
              </div>
            </form>
          ) : (
            <div className="space-y-2">
              {existingProjects.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <p>No existing projects found.</p>
                  <button
                    onClick={() => setMode('new')}
                    className="mt-2 text-blue-600 hover:text-blue-500"
                  >
                    Create your first project
                  </button>
                </div>
              ) : (
                existingProjects.map((project) => (
                  <button
                    key={project.id}
                    onClick={() => handleSelectProject(project)}
                    className="w-full p-4 text-left border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-colors"
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-medium text-gray-900">{project.name}</h3>
                        <p className="text-sm text-gray-500 font-mono">
                          {project.latitude.toFixed(4)}°N, {Math.abs(project.longitude).toFixed(4)}°W
                        </p>
                      </div>
                      {project.cached_capacity_factor && (
                        <span className="text-sm text-green-600 font-medium">
                          CF: {(project.cached_capacity_factor * 100).toFixed(1)}%
                        </span>
                      )}
                    </div>
                    {project.description && (
                      <p className="mt-1 text-sm text-gray-600 line-clamp-1">
                        {project.description}
                      </p>
                    )}
                  </button>
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
