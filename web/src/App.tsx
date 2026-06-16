import { type FormEvent, useState } from "react"
import { LoaderCircle, SendHorizontal } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
  FieldSet,
} from "@/components/ui/field"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Textarea } from "@/components/ui/textarea"
import { analyzeComment, type LabelKey, type LabelPrediction, type PredictResponse } from "@/lib/api"

const LABEL_COPY: Record<LabelKey, { title: string; description: string }> = {
  cleanliness: {
    title: "Cleanliness",
    description: "Signals that the review mentions hygiene or clean rooms.",
  },
  location: {
    title: "Location",
    description: "Signals about distance, neighborhood, transit, or central position.",
  },
  luxury: {
    title: "Luxury",
    description: "Signals about premium comfort, design, or high-end amenities.",
  },
  family_friendly: {
    title: "Family friendly",
    description: "Signals that the stay is suitable for families or children.",
  },
}

const LABEL_ORDER = Object.keys(LABEL_COPY) as LabelKey[]

const EXAMPLE_COMMENT =
  "The apartment was spotless, close to the city center, and perfect for our family trip."

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`
}

function LabelResultCard({
  labelKey,
  result,
}: {
  labelKey: LabelKey
  result?: LabelPrediction
}) {
  const copy = LABEL_COPY[labelKey]
  const probability = result ? Math.round(result.probability * 100) : 0

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>{copy.title}</CardTitle>
        <CardDescription>{copy.description}</CardDescription>
        <CardAction>
          {result ? (
            <Badge variant={result.prediction ? "default" : "secondary"}>
              {result.prediction ? "Yes" : "No"}
            </Badge>
          ) : (
            <Badge variant="outline">Pending</Badge>
          )}
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3 text-sm">
          <span className="text-muted-foreground">Probability</span>
          <span className="font-medium">{result ? formatPercent(result.probability) : "0%"}</span>
        </div>
        <Progress value={probability} />
      </CardContent>
    </Card>
  )
}

function RatingCard({ result }: { result?: PredictResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Visitor rating</CardTitle>
        <CardDescription>Synthetic regression target predicted from review text.</CardDescription>
      </CardHeader>
      <CardContent className="flex items-end justify-between gap-4">
        <div className="font-heading text-5xl font-medium leading-none">
          {result ? result.visitor_rating.toFixed(1) : "--"}
        </div>
        <div className="pb-1 text-sm text-muted-foreground">out of 5.0</div>
      </CardContent>
    </Card>
  )
}

function ModelInfoCard({ result }: { result?: PredictResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Loaded models</CardTitle>
        <CardDescription>Inference backend model metadata.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground">Classifier</span>
          <span className="font-medium">{result?.model_info.classifier ?? "Waiting for API response"}</span>
        </div>
        <Separator />
        <div className="flex flex-col gap-1">
          <span className="text-muted-foreground">Regressor</span>
          <span className="font-medium">{result?.model_info.regressor ?? "Waiting for API response"}</span>
        </div>
      </CardContent>
    </Card>
  )
}

function App() {
  const [comment, setComment] = useState(EXAMPLE_COMMENT)
  const [result, setResult] = useState<PredictResponse>()
  const [error, setError] = useState<string>()
  const [isPending, setIsPending] = useState(false)

  const canSubmit = comment.trim().length > 0 && !isPending

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!canSubmit) {
      return
    }

    setIsPending(true)
    setError(undefined)

    try {
      const prediction = await analyzeComment(comment.trim())
      setResult(prediction)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Prediction request failed.")
    } finally {
      setIsPending(false)
    }
  }

  return (
    <main className="min-h-svh bg-background text-foreground">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 py-6 md:px-8 md:py-10">
        <header className="flex flex-col gap-3">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div className="flex flex-col gap-2">
              <h1 className="font-heading text-3xl font-medium leading-tight md:text-4xl">
                Turismy review analyzer
              </h1>
              <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
                Local multi-label classification and visitor rating regression for tourism reviews.
              </p>
            </div>
            <Badge variant="outline">FastAPI + React</Badge>
          </div>
          <Separator />
        </header>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
          <Card>
            <CardHeader>
              <CardTitle>Review input</CardTitle>
              <CardDescription>Analyze one guest comment through the local inference API.</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                <FieldSet>
                  <FieldGroup>
                    <Field data-invalid={Boolean(error)}>
                      <FieldLabel htmlFor="comment">Comment</FieldLabel>
                      <Textarea
                        id="comment"
                        value={comment}
                        onChange={(event) => setComment(event.target.value)}
                        aria-invalid={Boolean(error)}
                        rows={9}
                      />
                      <FieldDescription>
                        The backend expects English review text similar to the training dataset.
                      </FieldDescription>
                    </Field>
                  </FieldGroup>
                </FieldSet>

                {error ? (
                  <Alert variant="destructive">
                    <AlertTitle>Prediction failed</AlertTitle>
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                ) : null}

                <div className="flex flex-col gap-2 sm:flex-row">
                  <Button type="submit" disabled={!canSubmit} size="lg">
                    {isPending ? (
                      <LoaderCircle data-icon="inline-start" className="animate-spin" />
                    ) : (
                      <SendHorizontal data-icon="inline-start" />
                    )}
                    {isPending ? "Analyzing" : "Analyze review"}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="lg"
                    onClick={() => {
                      setComment(EXAMPLE_COMMENT)
                      setError(undefined)
                    }}
                  >
                    Reset example
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          <div className="grid gap-4 sm:grid-cols-2">
            {LABEL_ORDER.map((labelKey) => (
              <LabelResultCard key={labelKey} labelKey={labelKey} result={result?.labels[labelKey]} />
            ))}
          </div>
        </section>

        <section className="grid gap-6 lg:grid-cols-[minmax(0,0.55fr)_minmax(0,1fr)]">
          <RatingCard result={result} />
          <ModelInfoCard result={result} />
        </section>
      </div>
    </main>
  )
}

export default App
