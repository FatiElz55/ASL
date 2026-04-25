export const metadata = {
  title: "ASL Detector",
  description: "Browser-based ASL sign detector",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
          background: "#0b1020",
          color: "#e8eaf2",
        }}
      >
        {children}
      </body>
    </html>
  );
}

