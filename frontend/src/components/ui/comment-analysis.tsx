
import { useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Loader2,
  MessageSquare,
  ShieldAlert,
  ShieldCheck,
} from "lucide-react";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type ThresholdAssessment = {
  status: string;
  threshold: number | null;
  flag: boolean | null;
};

type SafetyAnalysis = {
  safety_scores: Record<string, number>;
  threshold_assessments: Record<string, ThresholdAssessment>;
  qualified_review_signals: string[];
  qualified_review_signal_present: boolean;
  moderation_action: string;
};

type EngagementAnalysis = {
  predicted_intent: string;
  max_probability: number;
  second_probability: number;
  confidence_margin: number;
  class_probabilities: Record<string, number>;
  requires_review: boolean;
  review_reason: string;
  model_version: string;
};

type AnalysisResult = {
  analysis: {
    safety_analysis: SafetyAnalysis;
    engagement_analysis: EngagementAnalysis;
    requires_human_review: boolean;
    review_reasons: string[];
    moderation_action: string;
    analysis_version: string;
  };
  routing: {
    route: string;
    priority: string;
    reason: string;
    requires_human_approval: boolean;
    automatic_action_allowed: boolean;
    routing_version: string;
  };
  response_generation: {
    status: string;
    response: string | null;
    reason: string;
    requires_human_approval: boolean;
    automatic_action_allowed: boolean;
  };
  review: {
    review_id: string;
    review_status: string;
    review_priority: string;
  };
};

type ApiResponse = {
  success: boolean;
  message: string;
  data?: AnalysisResult;
  error?: string | null;
};

function formatLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function percentage(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function CommentAnalysis() {
  const [comment, setComment] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleAnalyze(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!comment.trim()) {
      setError("Please enter a comment before analyzing.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/comments/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          comment_text: comment.trim(),
        }),
      });

      const payload = (await response.json()) as ApiResponse;

      if (!response.ok || !payload.success || !payload.data) {
        throw new Error(
          payload.message || payload.error || "Comment analysis failed."
        );
      }

      setResult(payload.data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the moderation API."
      );
    } finally {
      setLoading(false);
    }
  }

  const safety = result?.analysis.safety_analysis;
  const engagement = result?.analysis.engagement_analysis;
  const routing = result?.routing;
  const responseGeneration = result?.response_generation;

  return (
    <div className="space-y-8">
      {/* Page heading */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          Comment Analysis
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Analyze comments using MODERAI safety and engagement models.
        </p>
      </div>

      {/* Comment input */}
      <section className="rounded-xl border border-border bg-card p-6 shadow-sm">
        <div className="mb-5 flex items-center gap-3">
          <div className="rounded-lg bg-indigo-50 p-2.5 text-indigo-600">
            <MessageSquare className="size-5" />
          </div>
          <div>
            <h2 className="font-semibold">Enter a comment</h2>
            <p className="text-xs text-muted-foreground">
              Submit text for moderation analysis.
            </p>
          </div>
        </div>

        <form onSubmit={handleAnalyze} className="space-y-4">
          <textarea
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            placeholder="Paste or type a comment here..."
            rows={5}
            maxLength={10000}
            disabled={loading}
            className="w-full resize-y rounded-lg border border-border bg-background p-4 text-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-100 disabled:opacity-60"
          />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              {comment.length} / 10,000 characters
            </p>

            <button
              type="submit"
              disabled={loading || !comment.trim()}
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <ShieldCheck className="size-4" />
                  Analyze Comment
                </>
              )}
            </button>
          </div>
        </form>

        {error && (
          <div
            role="alert"
            className="mt-4 flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700"
          >
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <p>{error}</p>
          </div>
        )}
      </section>

      {/* Analysis results */}
      {result && safety && engagement && routing && responseGeneration && (
        <>
          {/* Routing result */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-sm text-muted-foreground">
                  Recommended moderation route
                </p>
                <h2 className="mt-1 text-xl font-bold">
                  {formatLabel(routing.route)}
                </h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  {routing.reason}
                </p>
              </div>

              <span className="rounded-full bg-amber-50 px-3 py-1.5 text-sm font-semibold text-amber-700">
                {formatLabel(routing.priority)} Priority
              </span>
            </div>

            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <div className="rounded-lg bg-muted/50 p-4">
                <p className="text-xs text-muted-foreground">
                  Human approval
                </p>
                <p className="mt-1 font-semibold">
                  {routing.requires_human_approval ? "Required" : "Not required"}
                </p>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <p className="text-xs text-muted-foreground">
                  Automatic action
                </p>
                <p className="mt-1 font-semibold">
                  {routing.automatic_action_allowed ? "Allowed" : "Not allowed"}
                </p>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <p className="text-xs text-muted-foreground">Review status</p>
                <p className="mt-1 font-semibold">
                  {formatLabel(result.review.review_status)}
                </p>
              </div>
            </div>
          </section>

          {/* Safety analysis */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <div className="mb-6 flex items-center gap-3">
              <div className="rounded-lg bg-red-50 p-2.5 text-red-600">
                <ShieldAlert className="size-5" />
              </div>
              <div>
                <h2 className="font-semibold">Safety Analysis</h2>
                <p className="text-xs text-muted-foreground">
                  Six independent safety classifier scores
                </p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {Object.entries(safety.safety_scores).map(([name, score]) => {
                const assessment = safety.threshold_assessments[name];

                return (
                  <div
                    key={name}
                    className="rounded-lg border border-border p-4"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium">
                        {formatLabel(name)}
                      </p>
                      <span className="text-sm font-bold tabular-nums">
                        {percentage(score)}
                      </span>
                    </div>

                    <div className="mt-3 h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-indigo-600"
                        style={{
                          width: `${Math.min(score * 100, 100)}%`,
                        }}
                      />
                    </div>

                    <div className="mt-3 flex flex-wrap justify-between gap-2 text-xs">
                      <span className="text-muted-foreground">
                        Threshold:{" "}
                        {assessment?.threshold == null
                          ? "Not calibrated"
                          : percentage(assessment.threshold)}
                      </span>

                      <span
                        className={
                          assessment?.flag === true
                            ? "font-semibold text-red-600"
                            : assessment?.flag === false
                              ? "font-semibold text-green-600"
                              : "text-muted-foreground"
                        }
                      >
                        {assessment?.flag === true
                          ? "Flagged"
                          : assessment?.flag === false
                            ? "Not flagged"
                            : "Not assessed"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-5 rounded-lg bg-muted/50 p-4">
              <p className="text-sm font-medium">
                Qualified review signals
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                {safety.qualified_review_signals.length > 0
                  ? safety.qualified_review_signals
                      .map(formatLabel)
                      .join(", ")
                  : "No qualified safety review signals."}
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                Moderation action: {formatLabel(safety.moderation_action)}
              </p>
            </div>
          </section>

          {/* Engagement analysis */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h2 className="font-semibold">Engagement Analysis</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Independent engagement intent prediction
            </p>

            <div className="mt-5 grid gap-4 sm:grid-cols-3">
              <div className="rounded-lg bg-indigo-50 p-4">
                <p className="text-xs text-indigo-700">Predicted intent</p>
                <p className="mt-1 text-xl font-bold text-indigo-700">
                  {formatLabel(engagement.predicted_intent)}
                </p>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <p className="text-xs text-muted-foreground">
                  Maximum probability
                </p>
                <p className="mt-1 text-xl font-bold">
                  {percentage(engagement.max_probability)}
                </p>
              </div>

              <div className="rounded-lg bg-muted/50 p-4">
                <p className="text-xs text-muted-foreground">
                  Confidence margin
                </p>
                <p className="mt-1 text-xl font-bold">
                  {percentage(engagement.confidence_margin)}
                </p>
              </div>
            </div>

            <h3 className="mb-4 mt-7 text-sm font-semibold">
              Class probabilities
            </h3>

            <div className="space-y-4">
              {Object.entries(engagement.class_probabilities).map(
                ([name, probability]) => (
                  <div key={name}>
                    <div className="mb-1.5 flex justify-between text-sm">
                      <span>{formatLabel(name)}</span>
                      <span className="font-semibold tabular-nums">
                        {percentage(probability)}
                      </span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-indigo-500"
                        style={{
                          width: `${Math.min(probability * 100, 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                )
              )}
            </div>

            <div className="mt-6 flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4">
              <Clock3 className="mt-0.5 size-5 shrink-0 text-amber-700" />
              <div>
                <p className="text-sm font-semibold text-amber-800">
                  Human confirmation required
                </p>
                <p className="mt-1 text-xs leading-5 text-amber-800">
                  {engagement.review_reason}
                </p>
              </div>
            </div>

            <p className="mt-3 text-xs text-muted-foreground">
              Model: {engagement.model_version}
            </p>
          </section>

          {/* Response generation */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <div className="flex items-center gap-3">
              <CheckCircle2 className="size-5 text-indigo-600" />
              <h2 className="font-semibold">Response Generation</h2>
            </div>

            <p className="mt-4 text-sm font-semibold">
              Status: {formatLabel(responseGeneration.status)}
            </p>

            <p className="mt-2 text-sm text-muted-foreground">
              {responseGeneration.response ??
                responseGeneration.reason}
            </p>
          </section>

          {/* Review reference */}
          <section className="rounded-xl border border-border bg-card p-6 shadow-sm">
            <h2 className="font-semibold">Human Review Reference</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Review ID
            </p>
            <p className="mt-1 break-all font-mono text-sm">
              {result.review.review_id}
            </p>
            <p className="mt-3 text-xs text-muted-foreground">
              Review priority: {formatLabel(result.review.review_priority)}
            </p>
          </section>
        </>
      )}
    </div>
  );
}

export default CommentAnalysis;
