'use client'

import { useEffect, useState } from 'react'
import { useAuth } from '@/hooks/useAuth'
import { Navbar } from '@/components/ui/Navbar'
import { reportsApi, SavedReport } from '@/lib/api'
import Link from 'next/link'

export default function ReportsPage() {
  const { user, loading: authLoading } = useAuth()
  const [reports, setReports] = useState<SavedReport[]>([])
  const [loading, setLoading] = useState(true)
  const [viewingReportId, setViewingReportId] = useState<string | null>(null)
  const [reportHtml, setReportHtml] = useState<string>('')
  const [deletingId, setDeletingId] = useState<string | null>(null)

  useEffect(() => {
    if (user) {
      loadReports()
    } else {
      setLoading(false)
    }
  }, [user])

  const loadReports = async () => {
    if (!user) return
    try {
      setLoading(true)
      const data = await reportsApi.getUserReports(user.id)
      setReports(data)
    } catch (err) {
      console.error('Failed to load reports:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleViewReport = async (reportId: string) => {
    try {
      const html = await reportsApi.getReportHtml(reportId)
      setReportHtml(html)
      setViewingReportId(reportId)
    } catch (err) {
      console.error('Failed to load report:', err)
      alert('Failed to load report')
    }
  }

  const handleDownloadReport = async (reportId: string, projectName: string) => {
    try {
      const html = await reportsApi.getReportHtml(reportId)
      const blob = new Blob([html], { type: 'text/html' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${projectName.replace(/\s+/g, '_')}_report.html`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Failed to download report:', err)
      alert('Failed to download report')
    }
  }

  const handleDeleteReport = async (reportId: string) => {
    if (!user || !confirm('Are you sure you want to delete this report?')) return
    
    try {
      setDeletingId(reportId)
      await reportsApi.deleteReport(reportId, user.id)
      setReports(reports.filter(r => r.id !== reportId))
    } catch (err) {
      console.error('Failed to delete report:', err)
      alert('Failed to delete report')
    } finally {
      setDeletingId(null)
    }
  }

  const handlePrintReport = () => {
    const printWindow = window.open('', '_blank')
    if (printWindow) {
      printWindow.document.write(reportHtml)
      printWindow.document.close()
      printWindow.print()
    }
  }

  if (authLoading || loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="max-w-7xl mx-auto py-12 px-4">
          <div className="animate-pulse space-y-4">
            <div className="h-8 bg-gray-200 rounded w-1/4"></div>
            <div className="h-64 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="max-w-7xl mx-auto py-12 px-4 text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Reports</h1>
          <p className="text-gray-600 mb-6">Please sign in to view your reports.</p>
          <Link
            href="/login"
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Sign In
          </Link>
        </div>
      </div>
    )
  }

  // Report viewer modal
  if (viewingReportId) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="max-w-7xl mx-auto py-6 px-4">
          <div className="flex justify-between items-center mb-4">
            <button
              onClick={() => setViewingReportId(null)}
              className="flex items-center gap-2 text-blue-600 hover:text-blue-800"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Back to Reports
            </button>
            <div className="flex gap-2">
              <button
                onClick={handlePrintReport}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
                </svg>
                Print / Save as PDF
              </button>
            </div>
          </div>
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <iframe
              srcDoc={reportHtml}
              className="w-full h-[calc(100vh-200px)]"
              title="Report"
            />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Reports</h1>
          <span className="text-sm text-gray-500">{reports.length} report{reports.length !== 1 ? 's' : ''}</span>
        </div>

        {reports.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">No reports yet</h3>
            <p className="mt-1 text-sm text-gray-500">
              Generate a report from the Analytics page to see it here.
            </p>
            <div className="mt-6">
              <Link
                href="/dashboard"
                className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
              >
                Go to Dashboard
              </Link>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Report
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date Range
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Summary
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {reports.map((report) => (
                  <tr key={report.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {report.report_type.charAt(0).toUpperCase() + report.report_type.slice(1)} Report
                      </div>
                      <div className="text-sm text-gray-500">
                        {(report.parameters as { iso_region?: string })?.iso_region || 'N/A'}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {(report.parameters as { start_date?: string })?.start_date} to{' '}
                      {(report.parameters as { end_date?: string })?.end_date}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">
                        {(report.summary.total_energy_mwh / 1000).toFixed(1)} GWh
                      </div>
                      <div className="text-sm text-gray-500">
                        ${(report.summary.total_revenue / 1000).toFixed(0)}K revenue
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(report.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                      <button
                        onClick={() => handleViewReport(report.id)}
                        className="text-blue-600 hover:text-blue-900 mr-3"
                      >
                        View
                      </button>
                      <button
                        onClick={() => handleDownloadReport(report.id, `Report_${report.id.slice(0, 8)}`)}
                        className="text-gray-600 hover:text-gray-900 mr-3"
                      >
                        Download
                      </button>
                      <button
                        onClick={() => handleDeleteReport(report.id)}
                        disabled={deletingId === report.id}
                        className="text-red-600 hover:text-red-900"
                      >
                        {deletingId === report.id ? 'Deleting...' : 'Delete'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
