FROM golang:1.21-alpine

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git

# Copy source code
COPY . .

# Initialize Go module if needed
RUN go mod init device-service || true
RUN go get github.com/gorilla/mux
RUN go mod tidy

# Run the source code directly
CMD ["go", "run", "main.go"]