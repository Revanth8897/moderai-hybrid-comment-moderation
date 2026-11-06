
import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Bell,
  ClipboardList,
  LayoutDashboard,
  Loader2,
  LockKeyhole,
  LogOut,
  Mail,
  MessageSquare,
  Search,
  Settings,
  ShieldCheck,
} from "lucide-react";

import Dashboard from "./components/ui/dashboard";
import CommentAnalysis from "./components/ui/comment-analysis";
import ReviewQueue from "./components/ui/review-queue";
import {
  getAccessToken,
  getCurrentUser,
  login,
  logout,
  type UserProfile,
} from "./lib/auth";

const navigationItems = [
  { label: "Dashboard", icon: LayoutDashboard },
  { label: "Comment Analysis", icon: MessageSquare },
  { label: "Review Queue", icon: ClipboardList },
  { label: "Activity History", icon: Activity },
  { label: "Analytics", icon: BarChart3 },
];

function LoginScreen({
  onLogin,
}: {
  onLogin: (email: string, password: string) => Promise<void>;
}) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);

    try {
      await onLogin(email, password);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to sign in.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex size-14 items-center justify-center rounded-2xl bg-primary shadow-lg">
            <ShieldCheck className="size-8 text-white" />
          </div>

          <h1 className="text-3xl font-bold tracking-tight">MODERAI</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Intelligent Comment Moderation
          </p>
        </div>

        <div className="rounded-2xl border border-border bg-card p-8 shadow-sm">
          <div className="mb-7">
            <h2 className="text-xl font-semibold">Welcome back</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Sign in to access your moderation workspace.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label
                htmlFor="email"
                className="mb-2 block text-sm font-medium"
              >
                Email address
              </label>

              <div className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 focus-within:ring-2 focus-within:ring-primary">
                <Mail className="size-4 text-muted-foreground" />

                <input
                  id="email"
                  type="email"
                  autoComplete="username"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="reviewer@example.com"
                  className="h-11 w-full bg-transparent text-sm outline-none"
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="password"
                className="mb-2 block text-sm font-medium"
              >
                Password
              </label>

              <div className="flex items-center gap-3 rounded-lg border border-border bg-background px-3 focus-within:ring-2 focus-within:ring-primary">
                <LockKeyhole className="size-4 text-muted-foreground" />

                <input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="Enter your password"
                  className="h-11 w-full bg-transparent text-sm outline-none"
                />
              </div>
            </div>

            {error && (
              <div
                role="alert"
                className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700"
              >
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 text-sm font-semibold text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                "Sign in"
              )}
            </button>
          </form>

          <p className="mt-6 text-center text-xs text-muted-foreground">
            Secure access for authorized moderators and administrators.
          </p>
        </div>

        <p className="mt-6 text-center text-xs text-muted-foreground">
          MODERAI · Human-in-the-loop moderation
        </p>
      </div>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState("Dashboard");
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function validateSession() {
      if (!getAccessToken()) {
        setCheckingSession(false);
        return;
      }

      try {
        const user = await getCurrentUser();

        if (!cancelled) {
          setCurrentUser(user);
        }
      } catch {
        if (!cancelled) {
          setCurrentUser(null);
        }
      } finally {
        if (!cancelled) {
          setCheckingSession(false);
        }
      }
    }

    void validateSession();

    return () => {
      cancelled = true;
    };
  }, []);

  async function handleLogin(email: string, password: string) {
    const user = await login(email, password);
    setCurrentUser(user);
    setActivePage("Dashboard");
  }

  function handleLogout() {
    logout();
    setCurrentUser(null);
    setActivePage("Dashboard");
  }

  if (checkingSession) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="size-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">
            Validating your session...
          </p>
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return <LoginScreen onLogin={handleLogin} />;
  }

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 flex w-64 flex-col bg-sidebar text-sidebar-foreground">
        {/* Brand */}
        <div className="flex h-20 items-center gap-3 border-b border-sidebar-border px-6">
          <div className="flex size-10 items-center justify-center rounded-xl bg-primary">
            <ShieldCheck className="size-6 text-white" />
          </div>

          <div>
            <h1 className="text-lg font-bold tracking-wide">MODERAI</h1>
            <p className="text-xs text-slate-400">
              Intelligent Moderation
            </p>
          </div>
        </div>

        {/* Workspace */}
        <div className="px-5 pb-3 pt-7">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Workspace
          </p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-3">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.label;

            return (
              <button
                key={item.label}
                type="button"
                onClick={() => setActivePage(item.label)}
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-left text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "text-slate-300 hover:bg-slate-700 hover:text-white"
                }`}
              >
                <Icon className="size-5" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Settings */}
        <div className="border-t border-sidebar-border p-3">
          <button
            type="button"
            onClick={() => setActivePage("Settings")}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium transition-colors ${
              activePage === "Settings"
                ? "bg-indigo-600 text-white"
                : "text-slate-300 hover:bg-slate-700 hover:text-white"
            }`}
          >
            <Settings className="size-5" />
            <span>Settings</span>
          </button>
        </div>

        {/* User profile */}
        <div className="border-t border-sidebar-border p-4">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-full bg-indigo-500 text-sm font-semibold text-white">
              {currentUser.email.charAt(0).toUpperCase()}
            </div>

            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-semibold">
                {currentUser.email}
              </p>
              <p className="truncate text-xs text-slate-400">
                {currentUser.role}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={handleLogout}
            className="mt-4 flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-300 transition-colors hover:bg-slate-700 hover:text-white"
          >
            <LogOut className="size-4" />
            <span>Sign out</span>
          </button>
        </div>
      </aside>

      {/* Main application area */}
      <div className="ml-64 flex min-h-screen min-w-0 flex-1 flex-col">
        {/* Top navigation */}
        <header className="sticky top-0 z-30 flex h-20 items-center justify-between border-b border-border bg-card px-8">
          <div>
            <h2 className="text-xl font-semibold">{activePage}</h2>
            <p className="text-xs text-muted-foreground">
              Manage and monitor your moderation workspace
            </p>
          </div>

          <div className="flex items-center gap-5">
            {/* Search */}
            <div className="hidden w-64 items-center gap-2 rounded-lg border border-border bg-background px-3 py-2 md:flex">
              <Search className="size-4 text-muted-foreground" />
              <input
                type="search"
                placeholder="Search..."
                className="w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              />
            </div>

            {/* Notifications */}
            <button
              type="button"
              aria-label="Notifications"
              className="relative rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <Bell className="size-5" />
              <span className="absolute right-1.5 top-1.5 size-2 rounded-full bg-danger" />
            </button>

            {/* Profile */}
            <div className="flex items-center gap-2 border-l border-border pl-5">
              <div className="flex size-9 items-center justify-center rounded-full bg-indigo-100 text-sm font-semibold text-indigo-700">
                {currentUser.email.charAt(0).toUpperCase()}
              </div>

              <div className="hidden sm:block">
                <p className="text-sm font-semibold">
                  {currentUser.email}
                </p>
                <p className="text-xs text-muted-foreground">
                  {currentUser.role}
                </p>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 p-8">
          <div className="mx-auto max-w-7xl">
            {activePage === "Dashboard" ? (
              <Dashboard />
            ) : activePage === "Comment Analysis" ? (
              <CommentAnalysis />
            ) : activePage === "Review Queue" ? (
              <ReviewQueue />
            ) : (
              <div className="rounded-xl border border-border bg-card p-8 shadow-sm">
                <h2 className="text-lg font-semibold">{activePage}</h2>

                <p className="mt-2 text-sm text-muted-foreground">
                  This section will be implemented in the next development
                  steps.
                </p>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
