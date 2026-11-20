import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GridWise — building energy dashboard",
  description:
    "Forecast vs actual building electricity, anomaly flags and carbon-aware load shifting",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
