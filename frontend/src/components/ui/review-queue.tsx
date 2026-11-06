
import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Clock,
  Loader2,
  RefreshCw,
  ShieldCheck,
  X,
  Eye,
  Play,
} from "lucide-react";

import { getAuthHeaders } from "../../lib/auth";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const PAGE_SIZE = 10;

type Priority = "CRITICAL" | "HIGH" | "NORMAL";
type PriorityFilter = "ALL" | Priority;

interface GeneratedDraft {
  route?: string;
  status?: string;
  response?: {
    tone?: string;
    reply?: string;
    status?: string;
    generator?: string;
    is_simulated?: boolean;
    requires_human_approval?: boolean;
    automatic_action_allowed?: boolean;
  };
  requires_human_approval?: boolean;
  automatic_action_allowed?: boolean;
}

interface Review {
  review_id: string;
  comment: string;
  analysis: {
    requires_human_review?: boolean;
    safety_analysis?: {
      qualified_review_signals?: string[];
      [key: string]: unknown;
    };
    engagement_analysis?: {
      predicted_intent?: string;
      [key: string]: unknown;
    };
    generated_draft?: GeneratedDraft;
    [key: string]: unknown;
  };
  routing?: {
    route?: string;
    priority?: string;
    [key: string]: unknown;
  };
  review_priority: Priority;
  priority_reason: string | null;
  review_status: string;
  reviewer_id?: string | null;
  review_decision?: string | null;
  decision_reason?: string | null;
  created_at: string;
  updated_at?: string;
  automatic_action_allowed: boolean;
}

interface PendingReviewsData {
  total: number;
  skip: number;
  limit: number;
  priority: Priority | null;
  has_more: boolean;
  reviews: Review[];
}

interface APIResponse<T> {
  success: boolean;
  message: string;
  data: T;
  error?: string | null;
}

function priorityStyles(priority: Priority) {
  switch (priority) {
    case "CRITICAL":
      return "border-red-200 bg-red-50 text-red-700";
    case "HIGH":
      return "border-amber-200 bg-amber-50 text-amber-700";
    default:
      return "border-blue-200 bg-blue-50 text-blue-700";
  }
}

function formatDate(value: string): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

async function getErrorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail ?? body.message ?? `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

function ReviewQueue() {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [total, setTotal] = useState(0);
  const [skip, setSkip] = useState(0);
  const [hasMore, setHasMore] = useState(false);
  const [priority, setPriority] = useState<PriorityFilter>("ALL");

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedReview, setSelectedReview] = useState<Review | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [startingReview, setStartingReview] = useState(false);
  const [actionMessage, setActionMessage] = useState("");
  const [approvingReview, setApprovingReview] = useState(false);

  const fetchReviews = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setError("");

      try {
        const params = new URLSearchParams({
          skip: String(skip),
          limit: String(PAGE_SIZE),
        });

        if (priority !== "ALL") {
          params.set("priority", priority);
        }

        const response = await fetch(
          `${API_BASE_URL}/reviews/pending?${params.toString()}`,
          {
            method: "GET",
            headers: getAuthHeaders(),
            signal,
          },
        );

        if (!response.ok) {
          throw new Error(await getErrorMessage(response));
        }

        const result: APIResponse<PendingReviewsData> =
          await response.json();

        if (!result.success || !result.data) {
          throw new Error("Unable to retrieve pending reviews.");
        }

        setReviews(result.data.reviews);
        setTotal(result.data.total);
        setHasMore(result.data.has_more);
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          return;
        }

        setError(
          err instanceof Error
            ? err.message
            : "Something went wrong while loading reviews.",
        );
      } finally {
        if (!signal?.aborted) {
          setLoading(false);
        }
      }
    },
    [skip, priority],
  );

  useEffect(() => {
    const controller = new AbortController();

    void Promise.resolve().then(() => {
      if (!controller.signal.aborted) {
        return fetchReviews(controller.signal);
      }
    });

    return () => controller.abort();
  }, [fetchReviews]);

  async function openReview(reviewId: string) {
    setSelectedReview(null);
    setDetailError("");
    setActionMessage("");
    setDetailLoading(true);

    try {
      const response = await fetch(
        `${API_BASE_URL}/reviews/${encodeURIComponent(reviewId)}`,
        {
          method: "GET",
          headers: getAuthHeaders(),
        },
      );

      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const result: APIResponse<Review> = await response.json();

      if (!result.success || !result.data) {
        throw new Error("Unable to retrieve review details.");
      }

      setSelectedReview(result.data);
    } catch (err) {
      setDetailError(
        err instanceof Error
          ? err.message
          : "Unable to retrieve review details.",
      );
    } finally {
      setDetailLoading(false);
    }
  }

  async function handleStartReview() {
    if (!selectedReview) return;

    setStartingReview(true);
    setDetailError("");
    setActionMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/reviews/${encodeURIComponent(selectedReview.review_id)}/start`,
        {
          method: "POST",
          headers: {
            ...getAuthHeaders(),
            "Content-Type": "application/json",
          },
          body: JSON.stringify({}),
        },
      );

      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const result: APIResponse<Review> = await response.json();

      if (!result.success || !result.data) {
        throw new Error("Unable to start this review.");
      }

      setSelectedReview(result.data);
      setActionMessage("Review successfully started.");

      await fetchReviews();
    } catch (err) {
      setDetailError(
        err instanceof Error
          ? err.message
          : "Unable to start this review.",
      );
    } finally {
      setStartingReview(false);
    }
  }

  async function handleApproveForDraft() {
    if (!selectedReview) return;

    setApprovingReview(true);
    setDetailError("");
    setActionMessage("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/reviews/${encodeURIComponent(selectedReview.review_id)}/decision`,
        {
          method: "POST",
          headers: {
            ...getAuthHeaders(),
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            decision: "APPROVE_FOR_DRAFT",
            reason: "Approved for response draft generation after human review.",
          }),
        },
      );

      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const result: APIResponse<Review> = await response.json();

      if (!result.success || !result.data) {
        throw new Error("Unable to generate the response draft.");
      }

      setSelectedReview(result.data);
      setActionMessage(
        "Review approved. Response draft generated successfully.",
      );

      await fetchReviews();
    } catch (err) {
      setDetailError(
        err instanceof Error
          ? err.message
          : "Unable to approve this review.",
      );
    } finally {
      setApprovingReview(false);
    }
  }

  function handlePriorityChange(value: PriorityFilter) {
    setPriority(value);
    setSkip(0);
    setSelectedReview(null);
  }

  const firstItem = total === 0 ? 0 : skip + 1;
  const lastItem = Math.min(skip + reviews.length, total);

  const signals =
    selectedReview?.analysis?.safety_analysis?.qualified_review_signals ?? [];

  const engagementIntent =
    selectedReview?.analysis?.engagement_analysis?.predicted_intent ??
    "NOT AVAILABLE";

  return (
    <div className="space-y-6">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Pending Review Queue
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Review comments that require human confirmation.
          </p>
        </div>

        <button
          type="button"
          onClick={() => void fetchReviews()}
          disabled={loading}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium transition-colors hover:bg-muted disabled:opacity-50"
        >
          <RefreshCw className={`size-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-lg bg-indigo-50 text-indigo-600">
              <ShieldCheck className="size-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">
                Matching pending reviews
              </p>
              <p className="text-2xl font-bold">
                {loading ? "—" : total}
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-lg bg-amber-50 text-amber-600">
              <Clock className="size-5" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Current page</p>
              <p className="text-2xl font-bold">
                {loading ? "—" : reviews.length}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="font-semibold">Review records</h2>
          <p className="text-xs text-muted-foreground">
            Filter the queue by review priority.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          {(["ALL", "CRITICAL", "HIGH", "NORMAL"] as const).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => handlePriorityChange(value)}
              className={`rounded-lg border px-3 py-2 text-xs font-semibold transition-colors ${
                priority === value
                  ? "border-indigo-600 bg-indigo-600 text-white"
                  : "border-border bg-background text-muted-foreground hover:bg-muted"
              }`}
            >
              {value === "ALL" ? "All priorities" : value}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="flex items-center justify-center rounded-xl border border-border bg-card py-16">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="size-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Loading pending reviews...
            </p>
          </div>
        </div>
      )}

      {!loading && error && (
        <div
          role="alert"
          className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-5 text-red-700"
        >
          <AlertCircle className="mt-0.5 size-5 shrink-0" />
          <div>
            <p className="font-semibold">Unable to load the review queue</p>
            <p className="mt-1 text-sm">{error}</p>
            <button
              type="button"
              onClick={() => void fetchReviews()}
              className="mt-3 text-sm font-semibold underline"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {!loading && !error && reviews.length === 0 && (
        <div className="rounded-xl border border-border bg-card px-6 py-16 text-center">
          <ShieldCheck className="mx-auto size-10 text-muted-foreground" />
          <h3 className="mt-4 font-semibold">No pending reviews found</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            There are no pending reviews matching this priority filter.
          </p>
        </div>
      )}

      {!loading && !error && (
        <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(340px,0.85fr)]">
          <div className="space-y-4">
            {reviews.map((review) => {
              const cardSignals =
                review.analysis?.safety_analysis?.qualified_review_signals ?? [];

              const intent =
                review.analysis?.engagement_analysis?.predicted_intent ??
                "NOT AVAILABLE";

              const isSelected =
                selectedReview?.review_id === review.review_id;

              return (
                <article
                  key={review.review_id}
                  className={`rounded-xl border bg-card p-5 shadow-sm transition-all ${
                    isSelected
                      ? "border-indigo-500 ring-2 ring-indigo-100"
                      : "border-border hover:shadow-md"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span
                        className={`rounded-full border px-3 py-1 text-xs font-bold ${priorityStyles(
                          review.review_priority,
                        )}`}
                      >
                        {review.review_priority}
                      </span>
                      <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-700">
                        {review.review_status}
                      </span>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {formatDate(review.created_at)}
                    </span>
                  </div>

                  <p className="mt-4 whitespace-pre-wrap break-words text-sm leading-6">
                    {review.comment}
                  </p>

                  <div className="mt-4 grid gap-4 border-t border-border pt-4 sm:grid-cols-2">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Engagement intent
                      </p>
                      <p className="mt-1 text-sm font-semibold">{intent}</p>
                    </div>
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Safety signals
                      </p>
                      {cardSignals.length > 0 ? (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {cardSignals.map((signal) => (
                            <span
                              key={signal}
                              className="rounded-md bg-red-50 px-2 py-1 text-xs font-medium text-red-700"
                            >
                              {signal}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="mt-1 text-sm text-muted-foreground">
                          No qualified safety signals reported
                        </p>
                      )}
                    </div>
                  </div>

                  {review.priority_reason && (
                    <div className="mt-4 flex items-start gap-2 rounded-lg bg-muted/60 p-3">
                      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-600" />
                      <div>
                        <p className="text-xs font-semibold">Priority reason</p>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {review.priority_reason}
                        </p>
                      </div>
                    </div>
                  )}

                  <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4">
                    <span className="text-xs text-muted-foreground">
                      ID: {review.review_id}
                    </span>
                    <button
                      type="button"
                      onClick={() => void openReview(review.review_id)}
                      className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-indigo-700"
                    >
                      <Eye className="size-4" />
                      View details
                    </button>
                  </div>
                </article>
              );
            })}
          </div>

          <aside className="rounded-xl border border-border bg-card p-5 shadow-sm xl:sticky xl:top-5">
            <div className="flex items-center justify-between gap-3 border-b border-border pb-4">
              <div>
                <h2 className="font-semibold">Review details</h2>
                <p className="mt-1 text-xs text-muted-foreground">
                  Inspect a comment before taking action.
                </p>
              </div>
              {selectedReview && (
                <button
                  type="button"
                  onClick={() => {
                    setSelectedReview(null);
                    setDetailError("");
                    setActionMessage("");
                  }}
                  aria-label="Close review details"
                  className="rounded-lg p-2 text-muted-foreground hover:bg-muted"
                >
                  <X className="size-4" />
                </button>
              )}
            </div>

            {detailLoading && (
              <div className="flex flex-col items-center gap-3 py-12">
                <Loader2 className="size-7 animate-spin text-primary" />
                <p className="text-sm text-muted-foreground">
                  Loading review details...
                </p>
              </div>
            )}

            {!detailLoading && detailError && (
              <div role="alert" className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">
                {detailError}
              </div>
            )}

            {!detailLoading && !selectedReview && !detailError && (
              <div className="py-12 text-center">
                <Eye className="mx-auto size-9 text-muted-foreground" />
                <p className="mt-3 text-sm font-medium">
                  Select a review to inspect
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Full analysis and review actions will appear here.
                </p>
              </div>
            )}

            {!detailLoading && selectedReview && (
              <div className="space-y-5 pt-5">
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-full border px-3 py-1 text-xs font-bold ${priorityStyles(
                      selectedReview.review_priority,
                    )}`}
                  >
                    {selectedReview.review_priority}
                  </span>
                  <span className="rounded-full bg-muted px-3 py-1 text-xs font-semibold">
                    {selectedReview.review_status}
                  </span>
                </div>

                <section>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Original comment
                  </h3>
                  <p className="mt-2 whitespace-pre-wrap break-words rounded-lg bg-muted/50 p-3 text-sm leading-6">
                    {selectedReview.comment}
                  </p>
                </section>

                <section>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Safety analysis
                  </h3>
                  {signals.length > 0 ? (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {signals.map((signal) => (
                        <span
                          key={signal}
                          className="rounded-md bg-red-50 px-2 py-1 text-xs font-semibold text-red-700"
                        >
                          {signal}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-2 text-sm text-muted-foreground">
                      No qualified safety signals reported.
                    </p>
                  )}
                </section>

                <section>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Engagement analysis
                  </h3>
                  <p className="mt-2 text-sm font-semibold">
                    Predicted intent: {engagementIntent}
                  </p>
                </section>

                <section>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Routing information
                  </h3>
                  <p className="mt-2 text-sm">
                    Route: {String(selectedReview.routing?.route ?? "NOT AVAILABLE")}
                  </p>
                  <p className="mt-1 text-sm">
                    Priority: {String(selectedReview.routing?.priority ?? "NOT AVAILABLE")}
                  </p>
                  {selectedReview.priority_reason && (
                    <p className="mt-2 text-xs text-muted-foreground">
                      {selectedReview.priority_reason}
                    </p>
                  )}
                </section>

                {selectedReview.analysis.generated_draft && (
                  <section className="space-y-3 rounded-xl border border-indigo-200 bg-indigo-50/50 p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <h3 className="font-semibold text-indigo-900">
                        Generated Response Draft
                      </h3>

                      <span className="rounded-full border border-indigo-200 bg-white px-3 py-1 text-xs font-bold text-indigo-700">
                        {selectedReview.analysis.generated_draft.status ?? "DRAFT"}
                      </span>
                    </div>

                    <div className="rounded-lg border border-indigo-100 bg-white p-3">
                      <p className="whitespace-pre-wrap break-words text-sm leading-6 text-gray-800">
                        {selectedReview.analysis.generated_draft.response?.reply ??
                          "No generated reply available."}
                      </p>
                    </div>

                    <div className="space-y-1 text-xs text-muted-foreground">
                      <p>
                        Generator:{" "}
                        <strong>
                          {selectedReview.analysis.generated_draft.response?.generator ??
                            "Not available"}
                        </strong>
                      </p>

                      <p>
                        Tone:{" "}
                        <strong>
                          {selectedReview.analysis.generated_draft.response?.tone ??
                            "Not available"}
                        </strong>
                      </p>

                      <p>
                        Simulated:{" "}
                        <strong>
                          {selectedReview.analysis.generated_draft.response?.is_simulated
                            ? "Yes"
                            : "No"}
                        </strong>
                      </p>
                    </div>

                    <div className="border-t border-indigo-100 pt-3 text-xs">
                      <p className="font-semibold text-amber-800">
                        Human approval required:{" "}
                        {String(
                          selectedReview.analysis.generated_draft.requires_human_approval ??
                            selectedReview.analysis.generated_draft.response
                              ?.requires_human_approval ??
                            false,
                        )}
                      </p>

                      <p className="mt-1 font-semibold text-green-800">
                        Automatic action allowed:{" "}
                        {String(
                          selectedReview.analysis.generated_draft.automatic_action_allowed ??
                            selectedReview.analysis.generated_draft.response
                              ?.automatic_action_allowed ??
                            false,
                        )}
                      </p>
                    </div>
                  </section>
                )}

                <section className="space-y-2 border-t border-border pt-4">
                  <p className="text-xs text-muted-foreground">
                    Review ID: {selectedReview.review_id}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Created: {formatDate(selectedReview.created_at)}
                  </p>
                  {selectedReview.reviewer_id && (
                    <p className="text-xs text-muted-foreground">
                      Reviewer: {selectedReview.reviewer_id}
                    </p>
                  )}
                </section>

                {actionMessage && (
                  <div role="status" className="rounded-lg bg-green-50 p-3 text-sm text-green-700">
                    {actionMessage}
                  </div>
                )}

                {selectedReview.review_status === "PENDING" && (
                  <button
                    type="button"
                    onClick={() => void handleStartReview()}
                    disabled={startingReview}
                    className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {startingReview ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Play className="size-4" />
                    )}
                    {startingReview ? "Starting review..." : "Start review"}
                  </button>
                )}

                {selectedReview.review_status === "IN_REVIEW" && (
                  <div className="space-y-3">
                    <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-800">
                      This review is assigned and ready for a human decision.
                    </div>

                    <button
                      type="button"
                      onClick={() => void handleApproveForDraft()}
                      disabled={
                        approvingReview ||
                        signals.length > 0
                      }
                      className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-4 py-3 text-sm font-semibold text-white transition-colors hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {approvingReview ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <ShieldCheck className="size-4" />
                      )}

                      {approvingReview
                        ? "Generating draft..."
                        : "Approve & Generate Draft"}
                    </button>

                    {signals.length > 0 && (
                      <p className="text-xs text-red-600">
                        Draft generation is disabled because qualified safety signals were detected.
                      </p>
                    )}
                  </div>
                )}

                <p className="text-xs text-muted-foreground">
                  Automatic publishing, hiding, and deletion are not enabled.
                </p>
              </div>
            )}
          </aside>
        </div>
      )}

      {!loading && !error && total > 0 && (
        <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm text-muted-foreground">
            Showing {firstItem}–{lastItem} of {total} matching reviews
          </p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={skip === 0}
              onClick={() => setSkip(Math.max(0, skip - PAGE_SIZE))}
              className="inline-flex items-center gap-1 rounded-lg border border-border px-3 py-2 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ChevronLeft className="size-4" />
              Previous
            </button>
            <button
              type="button"
              disabled={!hasMore}
              onClick={() => setSkip(skip + PAGE_SIZE)}
              className="inline-flex items-center gap-1 rounded-lg border border-border px-3 py-2 text-sm font-medium hover:bg-muted disabled:cursor-not-allowed disabled:opacity-40"
            >
              Next
              <ChevronRight className="size-4" />
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default ReviewQueue;
