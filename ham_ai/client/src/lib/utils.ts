import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, '') // Remove special characters
    .replace(/[\s_-]+/g, '-') // Replace spaces and underscores with hyphens
    .replace(/^-+|-+$/g, '') // Remove leading/trailing hyphens
}

export function formatFrequency(frequencyMhz: number): string {
  if (frequencyMhz >= 1000) {
    return `${(frequencyMhz / 1000).toFixed(1)} GHz`
  } else {
    return `${frequencyMhz} MHz`
  }
}

export function formatCallsign(callsign: string): string {
  return callsign.toUpperCase()
}

export function parseGridSquare(grid: string): { lat: number; lon: number } {
  // Convert Maidenhead grid square to lat/lon coordinates
  if (grid.length < 4) {
    throw new Error('Invalid grid square format')
  }
  
  const field = grid.substring(0, 2).toUpperCase()
  const square = grid.substring(2, 4)
  
  const lonField = field.charCodeAt(0) - 'A'.charCodeAt(0)
  const latField = field.charCodeAt(1) - 'A'.charCodeAt(0)
  
  const lonSquare = parseInt(square.charAt(0))
  const latSquare = parseInt(square.charAt(1))
  
  const lon = (lonField * 20) + (lonSquare * 2) - 180 + 1
  const lat = (latField * 10) + latSquare - 90 + 0.5
  
  return { lat, lon }
}

export function calculateDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  // Calculate great circle distance in kilometers
  const R = 6371 // Earth's radius in km
  const dLat = (lat2 - lat1) * Math.PI / 180
  const dLon = (lon2 - lon1) * Math.PI / 180
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * 
    Math.sin(dLon/2) * Math.sin(dLon/2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a))
  return R * c
}
