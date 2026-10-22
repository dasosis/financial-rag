# syntax=docker/dockerfile:1

# A static binary on a distroless base: no shell, no package manager, CA
# certificates for the providers' TLS, and a non-root user.
FROM golang:1.27.0-alpine AS build
WORKDIR /src
# Dependencies in their own layer, so editing code doesn't re-download them
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gateway /gateway
USER nonroot:nonroot
# 8080: the OpenAI-compatible API; 9090: /metrics (internal only)
EXPOSE 8080 9090
ENTRYPOINT ["/gateway"]
