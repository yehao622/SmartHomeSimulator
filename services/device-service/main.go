package main

import (
    "encoding/json"
    "log"
    "net/http"
    "os"
    "github.com/gorilla/mux"
)

type Device struct {
    ID          string  `json:"id"`
    Name        string  `json:"name"`
    Type        string  `json:"type"`
    PowerLevel  float64 `json:"powerLevel"`
    Status      string  `json:"status"`
}

func main() {
    // Configure logging
    log.SetFlags(log.LstdFlags | log.Lshortfile)
    log.Printf("Starting device service...")

    router := mux.NewRouter()
    
    // Health check endpoint
    router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(map[string]string{
            "status": "healthy",
            "service": "device-service",
        })
    }).Methods("GET")

    // Device endpoints
    router.HandleFunc("/devices", getDevices).Methods("GET")
    router.HandleFunc("/devices/{id}", getDevice).Methods("GET")

    // Get port from environment or use default
    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    log.Printf("Device service listening on port %s", port)
    server := &http.Server{
        Addr:    ":" + port,
        Handler: router,
    }

    if err := server.ListenAndServe(); err != nil {
        log.Fatalf("Failed to start server: %v", err)
    }
}

func getDevices(w http.ResponseWriter, r *http.Request) {
    devices := []Device{
        {ID: "batt1", Name: "Battery Storage", Type: "battery", PowerLevel: 5.0, Status: "charging"},
        {ID: "solar1", Name: "Solar Panel", Type: "solar", PowerLevel: 3.2, Status: "active"},
        {ID: "ac1", Name: "Air Conditioner", Type: "appliance", PowerLevel: 1.5, Status: "active"},
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(devices)
}

func getDevice(w http.ResponseWriter, r *http.Request) {
    vars := mux.Vars(r)
    id := vars["id"]
    
    // Mock response - in production would fetch from database
    device := Device{ID: id, Name: "Device " + id, Type: "appliance", PowerLevel: 1.5, Status: "active"}
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(device)
}