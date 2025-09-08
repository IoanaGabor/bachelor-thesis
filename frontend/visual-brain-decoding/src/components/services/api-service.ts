function getIdToken(): string | null {
  if (typeof window === "undefined" || typeof window.sessionStorage === "undefined") {
    return null;
  }
  for (let i = 0; i < window.sessionStorage.length; i++) {
    const key = window.sessionStorage.key(i);
    if (key && key.startsWith("oidc.user:")) {
      try {
        const user = JSON.parse(window.sessionStorage.getItem(key) || "");
        if (user && user.id_token) {
          return user.id_token;
        }
      } catch (e) {
      }
    }
  }
  return null;
}

function getAuthHeaders(extraHeaders: Record<string, string> = {}) {
  const idToken = getIdToken();
  return idToken
    ? { ...extraHeaders, Authorization: `Bearer ${idToken}` }
    : { ...extraHeaders };
}

export const uploadRecording = async (formData: FormData) => {
  const headers = getAuthHeaders();

  const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}recordings/`, {
    method: "POST",
    body: formData,
    mode: "cors",
    headers,
  });

  if (!response.ok) {
    throw new Error("Failed to upload recording");
  }
  return await response.json();
};

export const fetchRecordings = async () => {
  try {
    const headers = getAuthHeaders({
      "Content-Type": "application/json",
    });
    console.log("Fetch Recordings Headers:", headers);

    const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}recordings/`, {
      method: "GET",
      headers,
    });

    if (!response.ok) {
      throw new Error("Failed to fetch recordings");
    }
    return await response.json();
  } catch (err) {
    console.log(err);
    throw new Error("Failed to fetch recordings");
  }
};

export const fetchRecordingById = async (id: string) => {
  const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}recordings/${id}`, {
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error("Recording not found");
  return response.json();
};

export const fetchReconstructionsForRecording = async (recordingId: string) => {
  try {
    const response = await fetch(
      `${import.meta.env.VITE_PUBLIC_ENDPOINT}recordings/${recordingId}/reconstructions`,
      {
        method: "GET",
        headers: getAuthHeaders({
          "Content-Type": "application/json",
        }),
      }
    );

    if (!response.ok) {
      throw new Error("Failed to fetch reconstructions");
    }
    return await response.json();
  } catch (err) {
    console.log(err);
    throw new Error("Failed to fetch reconstructions");
  }
};

export const fetchAllUsers = async () => {
  try {
    const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}admin/users`, {
      method: "GET",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch users");
    }

    return await response.json();
  } catch (err) {
    console.log(err);
    throw new Error("Failed to fetch users");
  }
};

export const requestReconstruction = async (id: string, numberOfSteps: number) => {
  try {
    const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}reconstructions/${id}`, {
      method: "POST",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
      }),
      body: JSON.stringify({
        number_of_steps: numberOfSteps,
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to request reconstruction");
    }

    return await response.json();
  } catch (err) {
    console.log(err);
    throw new Error("Failed to request reconstruction");
  }
};

export const fetchUserAttributes = async () => {
  try {
    const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}users/attributes`, {
      method: "GET",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
      }),
    });

    if (!response.ok) {
      if (response.status === 404) {
        throw new Error("User not found");
      }
      throw new Error("Failed to fetch user attributes");
    }

    return await response.json();
  } catch (err) {
    console.log(err);
    throw new Error("Failed to fetch user attributes");
  }
};

export const fetchStatistics = async () => {
  try {
    const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}reconstructions/statistics`, {
      method: "GET",
      headers: getAuthHeaders({
        "Content-Type": "application/json",
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch statistics");
    }
    return await response.json();
  } catch (err) {
    console.log(err);
    throw new Error("Failed to fetch statistics");
  }
};

export const deleteRecording = async (id: number) => {
  try {
    const response = await fetch(`${import.meta.env.VITE_PUBLIC_ENDPOINT}recordings/${id}`, {
      method: "DELETE",
      headers: getAuthHeaders(),
  });
  if (!response.ok) {
      throw new Error("Failed to delete recording");
    }

    return await response.json();
  } catch (err) {
    console.log(err);
    throw new Error("Failed to delete recording");
  }
};
