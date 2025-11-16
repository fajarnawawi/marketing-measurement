#!/bin/bash

###############################################################################
# Quick Generate - Marketing Measurement Synthetic Data
#
# Pre-configured commands for common data generation scenarios
#
# Usage:
#   ./quick_generate.sh <scenario>
#
# Scenarios:
#   test      - Tiny dataset for quick tests (1K campaigns, ~30K rows)
#   small     - Small dataset for development (10K campaigns, ~300K rows)
#   medium    - Medium dataset for learning (100K campaigns, ~3M rows)
#   large     - Large dataset for Redshift (1M campaigns, ~30M rows)
#   xlarge    - Extra large for Redshift (10M campaigns, ~300M rows)
#   full      - Full production simulation (100M campaigns, ~3B rows)
#   custom    - Interactive custom configuration
#
# Examples:
#   ./quick_generate.sh test
#   ./quick_generate.sh medium
#   ./quick_generate.sh custom
#
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default output directory
OUTPUT_BASE="./output"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GENERATOR_SCRIPT="$SCRIPT_DIR/generate_marketing_data.py"

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}"
    echo "================================================================================"
    echo "  Marketing Measurement Data Generator - Quick Generate"
    echo "================================================================================"
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_dependencies() {
    print_info "Checking dependencies..."

    # Check if Python script exists
    if [ ! -f "$GENERATOR_SCRIPT" ]; then
        print_error "Generator script not found: $GENERATOR_SCRIPT"
        exit 1
    fi

    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi

    # Check if required packages are installed
    python3 -c "import numpy, pandas, tqdm" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Required Python packages not found"
        echo "Please install: pip install numpy pandas tqdm"
        exit 1
    fi

    print_success "All dependencies found"
}

show_usage() {
    echo "Usage: $0 <scenario>"
    echo ""
    echo "Available scenarios:"
    echo "  test      - Tiny dataset for quick tests (1K campaigns, ~30K rows, ~10MB)"
    echo "  small     - Small dataset for development (10K campaigns, ~300K rows, ~100MB)"
    echo "  medium    - Medium dataset for learning (100K campaigns, ~3M rows, ~1GB)"
    echo "  large     - Large dataset for Redshift (1M campaigns, ~30M rows, ~10GB)"
    echo "  xlarge    - Extra large for Redshift (10M campaigns, ~300M rows, ~100GB)"
    echo "  full      - Full production simulation (100M campaigns, ~3B rows, ~1TB)"
    echo "  custom    - Interactive custom configuration"
    echo ""
    echo "Examples:"
    echo "  $0 test"
    echo "  $0 medium"
    echo "  $0 custom"
    echo ""
}

estimate_resources() {
    local scenario=$1

    case $scenario in
        test)
            echo "  Campaigns: 1,000"
            echo "  Total Rows: ~30,000"
            echo "  Disk Space: ~10 MB"
            echo "  Generation Time: ~30 seconds"
            ;;
        small)
            echo "  Campaigns: 10,000"
            echo "  Total Rows: ~300,000"
            echo "  Disk Space: ~100 MB"
            echo "  Generation Time: 2-5 minutes"
            ;;
        medium)
            echo "  Campaigns: 100,000"
            echo "  Total Rows: ~3,000,000"
            echo "  Disk Space: ~1 GB"
            echo "  Generation Time: 15-30 minutes"
            ;;
        large)
            echo "  Campaigns: 1,000,000"
            echo "  Total Rows: ~30,000,000"
            echo "  Disk Space: ~10 GB"
            echo "  Generation Time: 2-4 hours"
            ;;
        xlarge)
            echo "  Campaigns: 10,000,000"
            echo "  Total Rows: ~300,000,000"
            echo "  Disk Space: ~100 GB"
            echo "  Generation Time: 10-20 hours"
            ;;
        full)
            echo "  Campaigns: 100,000,000"
            echo "  Total Rows: ~3,000,000,000"
            echo "  Disk Space: ~1 TB"
            echo "  Generation Time: 50+ hours"
            ;;
    esac
}

confirm_generation() {
    local scenario=$1

    echo ""
    print_warning "You are about to generate: $scenario"
    echo ""
    estimate_resources $scenario
    echo ""

    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Generation cancelled"
        exit 0
    fi
}

###############################################################################
# Generation Scenarios
###############################################################################

generate_test() {
    print_info "Generating TINY dataset (test/debugging)..."

    python3 "$GENERATOR_SCRIPT" \
        --size tiny \
        --days 90 \
        --output-dir "$OUTPUT_BASE/test" \
        --format csv,sqlite \
        --seed 42

    print_success "Test dataset generated in $OUTPUT_BASE/test"
}

generate_small() {
    print_info "Generating SMALL dataset (development)..."

    python3 "$GENERATOR_SCRIPT" \
        --size small \
        --days 180 \
        --output-dir "$OUTPUT_BASE/small" \
        --format csv,sqlite \
        --seed 42

    print_success "Small dataset generated in $OUTPUT_BASE/small"
}

generate_medium() {
    print_info "Generating MEDIUM dataset (learning/training)..."

    python3 "$GENERATOR_SCRIPT" \
        --size medium \
        --days 365 \
        --output-dir "$OUTPUT_BASE/medium" \
        --format csv,sqlite \
        --seed 42 \
        --customers-multiplier 50 \
        --conversions-multiplier 100 \
        --touchpoints-multiplier 200

    print_success "Medium dataset generated in $OUTPUT_BASE/medium"
}

generate_large() {
    print_info "Generating LARGE dataset (Redshift testing)..."

    python3 "$GENERATOR_SCRIPT" \
        --size large \
        --days 365 \
        --output-dir "$OUTPUT_BASE/large" \
        --format csv,redshift \
        --compress \
        --seed 42 \
        --customers-multiplier 50 \
        --conversions-multiplier 100 \
        --touchpoints-multiplier 200

    print_success "Large dataset generated in $OUTPUT_BASE/large"
    print_info "DDL files available in $OUTPUT_BASE/large/redshift/"
}

generate_xlarge() {
    print_info "Generating EXTRA LARGE dataset (Redshift production)..."

    print_warning "This will take 10-20 hours and use ~100GB disk space"

    python3 "$GENERATOR_SCRIPT" \
        --size xlarge \
        --days 365 \
        --output-dir "$OUTPUT_BASE/xlarge" \
        --format csv,redshift \
        --compress \
        --seed 42 \
        --customers-multiplier 100 \
        --conversions-multiplier 200 \
        --touchpoints-multiplier 400

    print_success "Extra large dataset generated in $OUTPUT_BASE/xlarge"
    print_info "Upload to S3: aws s3 sync $OUTPUT_BASE/xlarge/redshift/ s3://your-bucket/data/"
}

generate_full() {
    print_info "Generating FULL dataset (production simulation)..."

    print_warning "This will take 50+ hours and use ~1TB disk space"
    print_warning "Recommended: Run on a dedicated server with ample resources"

    python3 "$GENERATOR_SCRIPT" \
        --size xxlarge \
        --days 730 \
        --output-dir "$OUTPUT_BASE/full" \
        --format csv,redshift \
        --compress \
        --seed 42 \
        --customers-multiplier 100 \
        --conversions-multiplier 200 \
        --touchpoints-multiplier 500

    print_success "Full dataset generated in $OUTPUT_BASE/full"
    print_info "Upload to S3: aws s3 sync $OUTPUT_BASE/full/redshift/ s3://your-bucket/data/"
}

generate_custom() {
    print_info "Custom dataset generation - Interactive mode"
    echo ""

    # Prompt for parameters
    read -p "Number of campaigns (or preset: tiny/small/medium/large): " campaigns
    read -p "Number of days (default: 365): " days
    days=${days:-365}

    read -p "Output directory (default: ./output/custom): " output_dir
    output_dir=${output_dir:-"$OUTPUT_BASE/custom"}

    read -p "Export formats (csv, sqlite, redshift - comma separated): " formats
    formats=${formats:-"csv,sqlite"}

    read -p "Random seed (default: 42): " seed
    seed=${seed:-42}

    read -p "Compress CSV files? (y/n): " compress
    compress_flag=""
    if [[ $compress =~ ^[Yy]$ ]]; then
        compress_flag="--compress"
    fi

    # Build command
    cmd="python3 \"$GENERATOR_SCRIPT\" --days $days --output-dir \"$output_dir\" --format $formats --seed $seed $compress_flag"

    # Check if campaigns is a preset or number
    if [[ $campaigns =~ ^[0-9]+$ ]]; then
        cmd="$cmd --rows $campaigns"
    else
        cmd="$cmd --size $campaigns"
    fi

    echo ""
    print_info "Command to execute:"
    echo "$cmd"
    echo ""

    read -p "Execute this command? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Generation cancelled"
        exit 0
    fi

    eval $cmd

    print_success "Custom dataset generated in $output_dir"
}

###############################################################################
# Post-Generation Actions
###############################################################################

show_post_generation_info() {
    local output_dir=$1

    echo ""
    print_header
    print_success "Generation Complete!"
    echo ""
    print_info "Output location: $output_dir"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. View data summary:"
    echo "   cat $output_dir/data_summary.txt"
    echo ""
    echo "2. Query SQLite database (if generated):"
    echo "   sqlite3 $output_dir/marketing_data.db"
    echo "   > SELECT COUNT(*) FROM campaigns;"
    echo ""
    echo "3. Explore CSV files:"
    echo "   head -n 10 $output_dir/csv/campaigns.csv"
    echo ""
    echo "4. Load to Redshift (if Redshift format generated):"
    echo "   aws s3 sync $output_dir/redshift/ s3://your-bucket/marketing-data/"
    echo "   psql -h your-redshift-cluster -f $output_dir/redshift/create_tables.sql"
    echo ""
}

###############################################################################
# Main Script
###############################################################################

main() {
    print_header

    # Check if scenario provided
    if [ $# -eq 0 ]; then
        print_error "No scenario specified"
        echo ""
        show_usage
        exit 1
    fi

    SCENARIO=$1

    # Check dependencies
    check_dependencies

    # Confirm for large datasets
    if [[ "$SCENARIO" == "large" ]] || [[ "$SCENARIO" == "xlarge" ]] || [[ "$SCENARIO" == "full" ]]; then
        confirm_generation $SCENARIO
    fi

    # Record start time
    START_TIME=$(date +%s)

    # Execute scenario
    case $SCENARIO in
        test)
            generate_test
            OUTPUT_DIR="$OUTPUT_BASE/test"
            ;;
        small)
            generate_small
            OUTPUT_DIR="$OUTPUT_BASE/small"
            ;;
        medium)
            generate_medium
            OUTPUT_DIR="$OUTPUT_BASE/medium"
            ;;
        large)
            generate_large
            OUTPUT_DIR="$OUTPUT_BASE/large"
            ;;
        xlarge)
            generate_xlarge
            OUTPUT_DIR="$OUTPUT_BASE/xlarge"
            ;;
        full)
            generate_full
            OUTPUT_DIR="$OUTPUT_BASE/full"
            ;;
        custom)
            generate_custom
            OUTPUT_DIR="$OUTPUT_BASE/custom"
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown scenario: $SCENARIO"
            echo ""
            show_usage
            exit 1
            ;;
    esac

    # Record end time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Convert duration to human-readable format
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    echo ""
    print_info "Generation time: ${HOURS}h ${MINUTES}m ${SECONDS}s"

    # Show post-generation info
    show_post_generation_info "$OUTPUT_DIR"
}

# Run main function
main "$@"
