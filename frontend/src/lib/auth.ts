
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

const TOKEN_KEY = "moderai_access_token";

export interface UserProfile {
  id: string;
  email: string;
  role: "ADMIN" | "REVIEWER";
  is_active: boolean;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
}

interface APIResponse<T> {
  success: boolean;
  message: string;
  data: T;
  error?: string | null;
}

export function getAccessToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function logout(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

export function getAuthHeaders(): HeadersInit {
  const token = getAccessToken();

  return token
    ? { Authorization: `Bearer ${token}` }
    : {};
}

async function getErrorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    return body.detail ?? body.message ?? "Something went wrong.";
  } catch {
    return `Request failed with status ${response.status}.`;
  }
}

export async function login(
  email: string,
  password: string,
): Promise<UserProfile> {
  const formData = new URLSearchParams();
  formData.set("username", email.trim().toLowerCase());
  formData.set("password", password);

  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  const tokenData: TokenResponse = await response.json();

  sessionStorage.setItem(TOKEN_KEY, tokenData.access_token);

  try {
    return await getCurrentUser();
  } catch (error) {
    logout();
    throw error;
  }
}

export async function getCurrentUser(): Promise<UserProfile> {
  const token = getAccessToken();

  if (!token) {
    throw new Error("Authentication token not found.");
  }

  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (!response.ok) {
    logout();
    throw new Error(await getErrorMessage(response));
  }

  const result: APIResponse<UserProfile> = await response.json();

  if (!result.success || !result.data?.is_active) {
    logout();
    throw new Error("Unable to validate the current user.");
  }

  return result.data;
}
