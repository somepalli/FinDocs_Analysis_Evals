ALTER TABLE intake_jobs DROP CONSTRAINT IF EXISTS intake_jobs_status_check;
ALTER TABLE intake_jobs ADD CONSTRAINT intake_jobs_status_check
CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled'));

DROP INDEX IF EXISTS intake_jobs_active_application;
CREATE UNIQUE INDEX intake_jobs_active_application
ON intake_jobs (application_id)
WHERE status IN ('queued', 'running');
