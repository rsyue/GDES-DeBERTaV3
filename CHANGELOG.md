# Changelog
All notable changes to fairseq2 are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.1.4.1] - Apr. 27, 2026
- Fixed gradient unscaling for proper gradient flow
- Added max grad norm parameter for gradient clipping
- Added prefetch_factor for better dataloader stability and reasonable default
