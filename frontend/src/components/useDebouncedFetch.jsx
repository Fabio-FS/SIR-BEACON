import { useState, useEffect, useRef } from "react";
import { fetchJSON } from "../api";

// Fetches whenever params change, debounced so dragging a slider
// only fires once it settles. Returns the latest result.
export default function useDebouncedFetch(path, params, delay = 80) {
  const [data, setData] = useState(null);
  const key = JSON.stringify(params);
  const timer = useRef(null);

  useEffect(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      fetchJSON(path, params).then(setData);
    }, delay);
    return () => clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key, delay]);

  return data;
}