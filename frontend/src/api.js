// Set this to your deployed backend URL. Falls back to localhost for dev.
export const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function fetchJSON(path, params) {
  const url = new URL(API_BASE + path);
  Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const res = await fetch(url);
  return res.json();
}