'use client'

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { projectsApi } from '@/lib/api'
import type { Project, ProjectFormData } from '@/lib/types'

export function useProjects(userId: string | undefined) {
  return useQuery({
    queryKey: ['projects', userId],
    queryFn: () => projectsApi.getAll(userId!),
    enabled: !!userId,
  })
}

export function useProject(id: string | undefined) {
  return useQuery({
    queryKey: ['project', id],
    queryFn: () => projectsApi.getById(id!),
    enabled: !!id,
  })
}

export function useCreateProject() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ userId, data }: { userId: string; data: ProjectFormData }) =>
      projectsApi.create(userId, data),
    onSuccess: (newProject) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      return newProject
    },
  })
}

export function useUpdateProject() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<ProjectFormData> }) =>
      projectsApi.update(id, data),
    onSuccess: (updatedProject) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      queryClient.invalidateQueries({ queryKey: ['project', updatedProject.id] })
    },
  })
}

export function useDeleteProject() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => projectsApi.delete(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
    },
  })
}

export function useUpdateProjectCache() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      id,
      results,
    }: {
      id: string
      results: {
        cached_annual_revenue: number
        cached_capacity_factor: number
        cached_iso: string | null
      }
    }) => projectsApi.updateCachedResults(id, results),
    onSuccess: (updatedProject) => {
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      queryClient.invalidateQueries({ queryKey: ['project', updatedProject.id] })
    },
  })
}
