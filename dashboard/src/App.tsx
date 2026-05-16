import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./lib/auth-context";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import type { JSX } from "react";

function Guard({ children }: { children: JSX.Element }) {
  const { state } = useAuth();
  if (state.loading) return <div style={{ padding: 24 }}>Loading…</div>;
  return state.loggedIn ? children : <Navigate to="/login" replace />;
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/" element={<Guard><Dashboard /></Guard>} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
