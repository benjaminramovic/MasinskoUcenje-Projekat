export type LabelKey = "cleanliness" | "location" | "luxury" | "family_friendly"

export type LabelPrediction = {
  prediction: boolean
  probability: number
}

export type PredictResponse = {
  labels: Record<LabelKey, LabelPrediction>
  visitor_rating: number
  model_info: {
    classifier: string
    regressor: string
  }
}

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000"

export async function analyzeComment(comment: string, signal?: AbortSignal): Promise<PredictResponse> {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ comment }),
    signal,
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = typeof payload?.detail === "string" ? payload.detail : "Prediction request failed."
    throw new Error(detail)
  }

  return response.json()
}
