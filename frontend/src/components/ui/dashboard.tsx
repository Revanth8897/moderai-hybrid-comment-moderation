
import {
  Activity,
  CheckCircle2,
  Clock3,
  MessageSquare,
  MoreHorizontal,
  ShieldAlert,
  TrendingUp,
} from "lucide-react";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const stats = [
  {
    title: "Total Comments",
    value: "12,845",
    change: "+12.8%",
    description: "Compared to last month",
    icon: MessageSquare,
    color: "text-blue-600",
    background: "bg-blue-50",
  },
  {
    title: "Approved for Draft",
    value: "8,492",
    change: "+8.2%",
    description: "Compared to last month",
    icon: CheckCircle2,
    color: "text-green-600",
    background: "bg-green-50",
  },
  {
    title: "Pending Review",
    value: "1,284",
    change: "Needs attention",
    description: "Awaiting human review",
    icon: Clock3,
    color: "text-amber-600",
    background: "bg-amber-50",
  },
  {
    title: "Safety Escalations",
    value: "156",
    change: "Priority queue",
    description: "Requires safety review",
    icon: ShieldAlert,
    color: "text-red-600",
    background: "bg-red-50",
  },
];

const activityData = [
  { day: "Mon", comments: 420, reviewed: 310 },
  { day: "Tue", comments: 580, reviewed: 430 },
  { day: "Wed", comments: 490, reviewed: 390 },
  { day: "Thu", comments: 720, reviewed: 540 },
  { day: "Fri", comments: 650, reviewed: 510 },
  { day: "Sat", comments: 850, reviewed: 670 },
  { day: "Sun", comments: 760, reviewed: 610 },
];

const recentComments = [
  {
    id: "CMT-1024",
    comment: "Really enjoyed using this product. Great experience!",
    category: "Positive",
    action: "Approve for Draft",
    status: "Reviewed",
    color: "bg-green-50 text-green-700",
  },
  {
    id: "CMT-1023",
    comment: "The delivery was delayed. Please improve your service.",
    category: "Critical",
    action: "Human Review",
    status: "Pending",
    color: "bg-amber-50 text-amber-700",
  },
  {
    id: "CMT-1022",
    comment: "This is completely unacceptable and offensive.",
    category: "Safety Signal",
    action: "Safety Escalation",
    status: "Priority",
    color: "bg-red-50 text-red-700",
  },
  {
    id: "CMT-1021",
    comment: "Can you share more details about this feature?",
    category: "Question",
    action: "Human Review",
    status: "Pending",
    color: "bg-blue-50 text-blue-700",
  },
];

function Dashboard() {
  return (
    <div className="space-y-8">
      {/* Page heading */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">
            Dashboard Overview
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Monitor comment moderation performance and review activity.
          </p>
        </div>

        <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm text-muted-foreground">
          <Activity className="size-4" />
          Sample dashboard data
        </div>
      </div>

      {/* KPI cards */}
      <section className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;

          return (
            <div
              key={stat.title}
              className="rounded-xl border border-border bg-card p-5 shadow-sm"
            >
              <div className="flex items-center justify-between">
                <div className={`rounded-lg p-2.5 ${stat.background}`}>
                  <Icon className={`size-5 ${stat.color}`} />
                </div>

                <button
                  type="button"
                  aria-label={`More options for ${stat.title}`}
                  className="rounded-md p-1 text-muted-foreground hover:bg-muted"
                >
                  <MoreHorizontal className="size-5" />
                </button>
              </div>

              <p className="mt-5 text-sm font-medium text-muted-foreground">
                {stat.title}
              </p>

              <div className="mt-1 flex flex-wrap items-baseline gap-2">
                <h2 className="text-2xl font-bold tracking-tight">
                  {stat.value}
                </h2>
                <span className={`text-xs font-medium ${stat.color}`}>
                  {stat.change}
                </span>
              </div>

              <p className="mt-2 text-xs text-muted-foreground">
                {stat.description}
              </p>
            </div>
          );
        })}
      </section>

      {/* Activity chart and summary */}
      <section className="grid grid-cols-1 gap-5 xl:grid-cols-3">
        <div className="rounded-xl border border-border bg-card p-6 shadow-sm xl:col-span-2">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="font-semibold">Moderation Activity</h2>
              <p className="mt-1 text-xs text-muted-foreground">
                Comments received and reviewed this week
              </p>
            </div>

            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-2">
                <span className="size-2.5 rounded-full bg-indigo-600" />
                Received
              </span>
              <span className="flex items-center gap-2">
                <span className="size-2.5 rounded-full bg-emerald-500" />
                Reviewed
              </span>
            </div>
          </div>

          <div className="mt-6 h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={activityData}
                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="receivedGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#4F46E5" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#4F46E5" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="reviewedGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10B981" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0} />
                  </linearGradient>
                </defs>

                <CartesianGrid
                  strokeDasharray="3 3"
                  vertical={false}
                  stroke="#E2E8F0"
                />

                <XAxis
                  dataKey="day"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#64748B", fontSize: 12 }}
                  dy={10}
                />

                <YAxis
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: "#64748B", fontSize: 12 }}
                />

                <Tooltip />

                <Area
                  type="monotone"
                  dataKey="comments"
                  name="Received"
                  stroke="#4F46E5"
                  strokeWidth={2}
                  fill="url(#receivedGradient)"
                />

                <Area
                  type="monotone"
                  dataKey="reviewed"
                  name="Reviewed"
                  stroke="#10B981"
                  strokeWidth={2}
                  fill="url(#reviewedGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Review summary */}
        <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
          <h2 className="font-semibold">Review Summary</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Current moderation distribution
          </p>

          <div className="mt-7 space-y-6">
            <div>
              <div className="mb-2 flex justify-between text-sm">
                <span className="text-muted-foreground">Reviewed</span>
                <span className="font-semibold">66%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full w-[66%] rounded-full bg-green-500" />
              </div>
            </div>

            <div>
              <div className="mb-2 flex justify-between text-sm">
                <span className="text-muted-foreground">Pending</span>
                <span className="font-semibold">24%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full w-[24%] rounded-full bg-amber-500" />
              </div>
            </div>

            <div>
              <div className="mb-2 flex justify-between text-sm">
                <span className="text-muted-foreground">Escalated</span>
                <span className="font-semibold">10%</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div className="h-full w-[10%] rounded-full bg-red-500" />
              </div>
            </div>
          </div>

          <div className="mt-8 rounded-lg bg-indigo-50 p-4">
            <div className="flex items-center gap-2 text-indigo-700">
              <TrendingUp className="size-4" />
              <span className="text-sm font-semibold">Moderation insight</span>
            </div>
            <p className="mt-2 text-xs leading-5 text-indigo-700">
              Review activity is trending upward in this sample dataset.
            </p>
          </div>
        </div>
      </section>

      {/* Recent comments */}
      <section className="overflow-hidden rounded-xl border border-border bg-card shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border p-6">
          <div>
            <h2 className="font-semibold">Recent Comments</h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Latest comments processed by the moderation system
            </p>
          </div>

          <button
            type="button"
            className="rounded-lg border border-border px-4 py-2 text-sm font-medium transition-colors hover:bg-muted"
          >
            View all comments
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead className="bg-muted/60 text-xs uppercase tracking-wide text-muted-foreground">
              <tr>
                <th className="px-6 py-4 font-semibold">Comment ID</th>
                <th className="px-6 py-4 font-semibold">Comment</th>
                <th className="px-6 py-4 font-semibold">Category</th>
                <th className="px-6 py-4 font-semibold">Suggested Action</th>
                <th className="px-6 py-4 font-semibold">Status</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-border">
              {recentComments.map((comment) => (
                <tr key={comment.id} className="transition-colors hover:bg-muted/30">
                  <td className="whitespace-nowrap px-6 py-4 font-medium">
                    {comment.id}
                  </td>

                  <td className="max-w-xs px-6 py-4 text-muted-foreground">
                    <p className="truncate">{comment.comment}</p>
                  </td>

                  <td className="whitespace-nowrap px-6 py-4">
                    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${comment.color}`}>
                      {comment.category}
                    </span>
                  </td>

                  <td className="whitespace-nowrap px-6 py-4 text-muted-foreground">
                    {comment.action}
                  </td>

                  <td className="whitespace-nowrap px-6 py-4">
                    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${comment.color}`}>
                      {comment.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default Dashboard;
