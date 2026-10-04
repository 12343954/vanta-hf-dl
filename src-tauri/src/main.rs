#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, RANGE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::process::Command;
use std::{
    collections::HashSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant},
};
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, TrayIconBuilder, TrayIconEvent},
    AppHandle, Emitter, Manager, State, WindowEvent,
};
use tokio::io::AsyncWriteExt;

const SETTINGS_KEY: &str = "settings";
const TOKEN_KEY: &str = "hf_access_token";
const FAVORITES_KEY: &str = "favorites";
const PART_SIZE: i64 = 32 * 1024 * 1024;
const MAX_PART_DOWNLOAD_ATTEMPTS: usize = 6;
const TRAY_ID: &str = "main-tray";
const TRAY_SHOW_ID: &str = "tray-show";
const TRAY_MINIMIZE_ID: &str = "tray-minimize";
const TRAY_QUIT_ID: &str = "tray-quit";

const KNOWN_MODEL_ROOTS: &[&str] = &[
    "checkpoints",
    "clip",
    "clip_vision",
    "configs",
    "controlnet",
    "diffusion_models",
    "embeddings",
    "gligen",
    "hypernetworks",
    "latent_upscale_models",
    "loras",
    "model_patches",
    "photomaker",
    "style_models",
    "text_encoders",
    "unet",
    "upscale_models",
    "vae",
    "vae_approx",
];

struct AppState {
    active_jobs: Mutex<HashSet<String>>,
}

#[derive(Debug, Serialize)]
struct BackendStatus {
    ready: bool,
    storage: &'static str,
    downloader: &'static str,
}

#[derive(Debug, Deserialize)]
struct TargetPreviewRequest {
    models_root: String,
    file_path: String,
    subfolder: String,
}

#[derive(Debug, Serialize)]
struct TargetPreview {
    target_path: String,
}

#[derive(Debug, Serialize)]
struct AppStorage {
    settings: serde_json::Value,
    access_token: String,
}

#[derive(Debug, Deserialize)]
struct SaveSettingsRequest {
    settings: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct SaveTokenRequest {
    access_token: String,
}

#[derive(Debug, Deserialize)]
struct RefreshDownloadSourcesRequest {
    hf_endpoint: String,
}

#[derive(Debug, Deserialize)]
struct PickFolderRequest {
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EnqueueDownloadsRequest {
    jobs: Vec<DownloadJob>,
    access_token: String,
    proxy: String,
    max_concurrent_parts: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct DownloadJob {
    id: String,
    repo_id: String,
    file_path: String,
    file_name: String,
    size: i64,
    downloaded_bytes: i64,
    status: String,
    source_url: String,
    target_path: String,
    created_at: String,
    updated_at: String,
    sha256: Option<String>,
    error: Option<String>,
    warning: Option<String>,
}

#[derive(Debug, Clone)]
struct DownloadPart {
    part_index: i64,
    start_byte: i64,
    end_byte: i64,
    downloaded_bytes: i64,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct DownloadProgressEvent {
    job_id: String,
    downloaded_bytes: i64,
    status: String,
    error: Option<String>,
    warning: Option<String>,
}

#[tauri::command]
fn backend_status() -> BackendStatus {
    BackendStatus {
        ready: true,
        storage: "sqlite",
        downloader: "rust-range-32mb",
    }
}

#[tauri::command]
fn load_app_storage() -> Result<AppStorage, String> {
    let connection = open_database()?;
    Ok(AppStorage {
        settings: read_value(&connection, SETTINGS_KEY)?
            .and_then(|value| serde_json::from_str(&value).ok())
            .unwrap_or_else(|| serde_json::json!({})),
        access_token: read_value(&connection, TOKEN_KEY)?.unwrap_or_default(),
    })
}

#[tauri::command]
fn save_app_settings(request: SaveSettingsRequest) -> Result<serde_json::Value, String> {
    let connection = open_database()?;
    write_value(&connection, SETTINGS_KEY, &request.settings.to_string())?;
    Ok(request.settings)
}

#[tauri::command]
fn save_access_token(request: SaveTokenRequest) -> Result<(), String> {
    let connection = open_database()?;
    if request.access_token.trim().is_empty() {
        connection
            .execute("delete from app_kv where key = ?1", [TOKEN_KEY])
            .map_err(|error| format!("failed to clear token: {error}"))?;
    } else {
        write_value(&connection, TOKEN_KEY, request.access_token.trim())?;
    }
    Ok(())
}

#[tauri::command]
fn load_favorites() -> Result<serde_json::Value, String> {
    let connection = open_database()?;
    Ok(read_value(&connection, FAVORITES_KEY)?
        .and_then(|value| serde_json::from_str(&value).ok())
        .unwrap_or_else(|| serde_json::json!([])))
}

#[tauri::command]
fn save_favorites(favorites: serde_json::Value) -> Result<serde_json::Value, String> {
    let connection = open_database()?;
    write_value(&connection, FAVORITES_KEY, &favorites.to_string())?;
    Ok(favorites)
}

#[tauri::command]
fn preview_download_target(request: TargetPreviewRequest) -> TargetPreview {
    TargetPreview {
        target_path: build_target_path(
            &request.models_root,
            &request.file_path,
            &request.subfolder,
        )
        .to_string_lossy()
        .to_string(),
    }
}

#[tauri::command]
fn refresh_download_sources(
    request: RefreshDownloadSourcesRequest,
) -> Result<Vec<DownloadJob>, String> {
    refresh_incomplete_download_sources(&request.hf_endpoint)?;
    read_download_jobs()
}

#[tauri::command]
async fn list_download_jobs() -> Result<Vec<DownloadJob>, String> {
    tauri::async_runtime::spawn_blocking(|| {
        scan_existing_completed_files()?;
        read_download_jobs()
    })
    .await
    .map_err(|error| format!("list jobs task failed: {error}"))?
}

#[tauri::command]
async fn list_active_download_jobs() -> Result<Vec<DownloadJob>, String> {
    tauri::async_runtime::spawn_blocking(read_active_download_jobs)
        .await
        .map_err(|error| format!("list active jobs task failed: {error}"))?
}

#[tauri::command]
async fn enqueue_downloads(
    app: AppHandle,
    state: State<'_, AppState>,
    request: EnqueueDownloadsRequest,
) -> Result<Vec<DownloadJob>, String> {
    let mut spawn_ids = Vec::new();
    {
        let connection = open_database()?;
        for job in &request.jobs {
            if target_file_is_complete(&job.target_path, job.size) {
                let mut completed_job = job.clone();
                completed_job.status = "completed".to_string();
                completed_job.downloaded_bytes = job.size;
                upsert_download_job(&connection, &completed_job)?;
                let _ = fs::remove_dir_all(parts_dir(&job.target_path));
                continue;
            }

            if existing_stable_job_for_target(&connection, job)?.is_some() {
                continue;
            }

            upsert_download_job(&connection, job)?;
            ensure_parts(&connection, job)?;
            spawn_ids.push(job.id.clone());
        }
    }

    for job_id in spawn_ids {
        spawn_download_job(
            app.clone(),
            state.inner(),
            job_id,
            request.access_token.clone(),
            request.proxy.clone(),
            request.max_concurrent_parts.unwrap_or(4).clamp(1, 8),
        );
    }

    read_download_jobs()
}

#[tauri::command]
async fn pause_download_job(job_id: String) -> Result<Vec<DownloadJob>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        set_job_status(&job_id, "paused", None)?;
        read_download_jobs()
    })
    .await
    .map_err(|error| format!("pause task failed: {error}"))?
}

#[tauri::command]
async fn resume_download_job(
    app: AppHandle,
    state: State<'_, AppState>,
    job_id: String,
    access_token: String,
    proxy: String,
    max_concurrent_parts: Option<usize>,
) -> Result<Vec<DownloadJob>, String> {
    let current_status = read_job_status(&job_id)?;
    if matches!(current_status.as_str(), "queued" | "downloading") {
        return read_download_jobs();
    }

    set_job_status(&job_id, "queued", None)?;
    spawn_download_job(
        app,
        state.inner(),
        job_id,
        access_token,
        proxy,
        max_concurrent_parts.unwrap_or(2).clamp(1, 8),
    );
    read_download_jobs()
}

#[tauri::command]
async fn cancel_download_job(job_id: String) -> Result<Vec<DownloadJob>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        set_job_status(&job_id, "canceled", None)?;
        read_download_jobs()
    })
    .await
    .map_err(|error| format!("cancel task failed: {error}"))?
}

#[tauri::command]
async fn remove_download_job(job_id: String) -> Result<Vec<DownloadJob>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let connection = open_database()?;
        connection
            .execute("delete from download_parts where job_id = ?1", [&job_id])
            .map_err(|error| format!("failed to remove parts: {error}"))?;
        connection
            .execute("delete from download_jobs where id = ?1", [&job_id])
            .map_err(|error| format!("failed to remove job: {error}"))?;
        read_download_jobs()
    })
    .await
    .map_err(|error| format!("remove task failed: {error}"))?
}

#[tauri::command]
fn pick_folder(request: PickFolderRequest) -> Result<Option<String>, String> {
    pick_folder_impl(request.title.as_deref())
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    open_path_impl(&path)
}

#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    open_url_impl(&url)
}

fn spawn_download_job(
    app: AppHandle,
    state: &AppState,
    job_id: String,
    access_token: String,
    proxy: String,
    max_concurrent_parts: usize,
) {
    {
        let mut active_jobs = state.active_jobs.lock().expect("active job lock poisoned");
        if !active_jobs.insert(job_id.clone()) {
            return;
        }
    }

    tauri::async_runtime::spawn(async move {
        let result = run_download_job(
            app.clone(),
            job_id.clone(),
            access_token,
            proxy,
            max_concurrent_parts,
        )
        .await;
        if let Err(error) = result {
            let _ = set_job_status(&job_id, "failed", Some(&error));
            emit_job_progress(&app, &job_id, "failed", Some(error));
        }
        let state = app.state::<AppState>();
        let mut active_jobs = state.active_jobs.lock().expect("active job lock poisoned");
        active_jobs.remove(&job_id);
    });
}

async fn run_download_job(
    app: AppHandle,
    job_id: String,
    access_token: String,
    proxy: String,
    max_concurrent_parts: usize,
) -> Result<(), String> {
    set_job_status(&job_id, "downloading", None)?;
    emit_job_progress(&app, &job_id, "downloading", None);

    let job = read_download_job(&job_id)?.ok_or_else(|| "download job not found".to_string())?;
    if complete_from_existing_file(&app, &job)? {
        return Ok(());
    }

    prepare_part_files(&job)?;
    let client = build_client(&proxy)?;
    let parts = read_incomplete_parts(&job_id)?;
    let mut stream = futures_util::stream::iter(parts)
        .map(|part| {
            let app = app.clone();
            let client = client.clone();
            let job = job.clone();
            let access_token = access_token.clone();
            async move { download_part_with_retries(app, client, job, part, access_token).await }
        })
        .buffer_unordered(max_concurrent_parts);

    while let Some(result) = stream.next().await {
        result?;
    }

    if complete_from_existing_file(&app, &job)? {
        return Ok(());
    }

    if matches!(
        read_job_status(&job_id)?.as_str(),
        "canceled" | "paused" | "completed"
    ) {
        return Ok(());
    }

    merge_parts(&job).await?;
    let warning = checksum_warning(&job).await?;
    set_job_completed(&job_id, job.size, warning.as_deref())?;
    emit_job_progress(&app, &job_id, "completed", None);
    Ok(())
}

async fn download_part_with_retries(
    app: AppHandle,
    client: reqwest::Client,
    job: DownloadJob,
    part: DownloadPart,
    access_token: String,
) -> Result<(), String> {
    let mut last_error = String::new();
    for attempt in 1..=MAX_PART_DOWNLOAD_ATTEMPTS {
        match download_part(
            app.clone(),
            client.clone(),
            job.clone(),
            part.clone(),
            access_token.clone(),
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(error) => {
                if matches!(
                    read_job_status(&job.id)?.as_str(),
                    "paused" | "canceled" | "completed"
                ) {
                    return Ok(());
                }
                last_error = format!(
                    "part {} attempt {attempt}/{MAX_PART_DOWNLOAD_ATTEMPTS} failed: {error}",
                    part.part_index
                );
                let downloaded = tokio::fs::metadata(part_path(&job.target_path, part.part_index))
                    .await
                    .map(|meta| meta.len() as i64)
                    .unwrap_or(0);
                set_part_progress(
                    &job.id,
                    part.part_index,
                    downloaded,
                    "queued",
                    Some(&last_error),
                )?;
                let total = update_job_downloaded(&job.id)?;
                emit_job_progress_with_total(
                    &app,
                    &job.id,
                    total,
                    "downloading",
                    Some(last_error.clone()),
                );
                if attempt < MAX_PART_DOWNLOAD_ATTEMPTS {
                    tokio::time::sleep(Duration::from_secs((attempt as u64).min(5))).await;
                }
            }
        }
    }

    Err(last_error)
}

async fn download_part(
    app: AppHandle,
    client: reqwest::Client,
    job: DownloadJob,
    part: DownloadPart,
    access_token: String,
) -> Result<(), String> {
    if complete_from_existing_file(&app, &job)? {
        return Ok(());
    }

    if matches!(
        read_job_status(&job.id)?.as_str(),
        "paused" | "canceled" | "completed"
    ) {
        return Ok(());
    }

    let part_path = part_path(&job.target_path, part.part_index);
    let part_len = part.end_byte - part.start_byte + 1;
    let disk_bytes = tokio::fs::metadata(&part_path)
        .await
        .map(|meta| meta.len() as i64)
        .unwrap_or(0);
    if disk_bytes > part_len {
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .open(&part_path)
            .await
            .map_err(|error| format!("failed to open oversized part file: {error}"))?;
        file.set_len(part_len as u64)
            .await
            .map_err(|error| format!("failed to trim oversized part file: {error}"))?;
    }
    let offset = disk_bytes.min(part_len);
    if offset != part.downloaded_bytes {
        set_part_progress(&job.id, part.part_index, offset, "queued", None)?;
    }
    if part.start_byte + offset > part.end_byte {
        set_part_completed(&job.id, part.part_index, offset)?;
        let total = update_job_downloaded(&job.id)?;
        emit_job_progress_with_total(&app, &job.id, total, "downloading", None);
        return Ok(());
    }

    let range_start = part.start_byte + offset;
    let range_header = format!("bytes={range_start}-{}", part.end_byte);
    let mut request = client.get(&job.source_url).header(RANGE, range_header);
    if !access_token.trim().is_empty() {
        request = request.header(AUTHORIZATION, format!("Bearer {}", access_token.trim()));
    }

    let response = request
        .send()
        .await
        .map_err(|error| format!("part {} request failed: {error}", part.part_index))?;
    if response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err(format!(
            "part {} range request failed with {}",
            part.part_index,
            response.status()
        ));
    }

    if let Some(parent) = part_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|error| format!("failed to create part directory: {error}"))?;
    }
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&part_path)
        .await
        .map_err(|error| format!("failed to open part file: {error}"))?;

    let mut downloaded = offset;
    let mut last_progress_flush = Instant::now();
    let mut body = response.bytes_stream();
    while let Some(chunk) = body.next().await {
        let status = read_job_status(&job.id)?;
        if complete_from_existing_file(&app, &job)? {
            return Ok(());
        }

        if matches!(status.as_str(), "paused" | "canceled" | "completed") {
            set_part_progress(&job.id, part.part_index, downloaded, &status, None)?;
            let total = update_job_downloaded(&job.id)?;
            emit_job_progress_with_total(&app, &job.id, total, &status, None);
            return Ok(());
        }

        let chunk = chunk.map_err(|error| {
            format!(
                "part {} stream failed after {downloaded}/{part_len} bytes: {error}",
                part.part_index
            )
        })?;
        file.write_all(&chunk)
            .await
            .map_err(|error| format!("failed to write part file: {error}"))?;
        downloaded += chunk.len() as i64;
        if last_progress_flush.elapsed() >= Duration::from_millis(500) {
            set_part_progress(&job.id, part.part_index, downloaded, "downloading", None)?;
            let total = update_job_downloaded(&job.id)?;
            emit_job_progress_with_total(&app, &job.id, total, "downloading", None);
            last_progress_flush = Instant::now();
        }
    }

    set_part_completed(&job.id, part.part_index, downloaded)?;
    let total = update_job_downloaded(&job.id)?;
    emit_job_progress_with_total(&app, &job.id, total, "downloading", None);
    Ok(())
}

fn build_client(proxy: &str) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().user_agent("Vanta/1.0");
    if !proxy.trim().is_empty() {
        builder = builder.proxy(
            reqwest::Proxy::all(proxy.trim()).map_err(|error| format!("invalid proxy: {error}"))?,
        );
    }
    builder
        .build()
        .map_err(|error| format!("failed to build http client: {error}"))
}

fn prepare_part_files(job: &DownloadJob) -> Result<(), String> {
    if let Some(parent) = Path::new(&job.target_path).parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create target directory: {error}"))?;
    }
    fs::create_dir_all(parts_dir(&job.target_path))
        .map_err(|error| format!("failed to create parts directory: {error}"))?;
    Ok(())
}

async fn merge_parts(job: &DownloadJob) -> Result<(), String> {
    let final_path = PathBuf::from(&job.target_path);
    let temp_path = final_path.with_extension(format!(
        "{}.part",
        final_path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("download")
    ));
    let mut output = tokio::fs::File::create(&temp_path)
        .await
        .map_err(|error| format!("failed to create target file: {error}"))?;
    let part_count = ((job.size + PART_SIZE - 1) / PART_SIZE).max(1);
    for index in 0..part_count {
        let bytes = tokio::fs::read(part_path(&job.target_path, index))
            .await
            .map_err(|error| format!("failed to read part {index}: {error}"))?;
        output
            .write_all(&bytes)
            .await
            .map_err(|error| format!("failed to merge part {index}: {error}"))?;
    }
    output
        .flush()
        .await
        .map_err(|error| format!("failed to flush target file: {error}"))?;
    drop(output);
    let merged_size = tokio::fs::metadata(&temp_path)
        .await
        .map_err(|error| format!("failed to inspect merged file: {error}"))?
        .len() as i64;
    if merged_size != job.size {
        return Err(format!(
            "size mismatch after merge: expected {}, got {merged_size}",
            job.size
        ));
    }
    tokio::fs::rename(&temp_path, &final_path)
        .await
        .map_err(|error| format!("failed to finalize target file: {error}"))?;
    let _ = tokio::fs::remove_dir_all(parts_dir(&job.target_path)).await;
    Ok(())
}

fn parts_dir(target_path: &str) -> PathBuf {
    PathBuf::from(format!("{target_path}.parts"))
}

fn part_path(target_path: &str, index: i64) -> PathBuf {
    parts_dir(target_path).join(format!("part-{index:06}"))
}

fn emit_job_progress(app: &AppHandle, job_id: &str, status: &str, error: Option<String>) {
    let downloaded_bytes = read_download_job(job_id)
        .ok()
        .flatten()
        .map(|job| job.downloaded_bytes)
        .unwrap_or(0);
    emit_job_progress_with_total(app, job_id, downloaded_bytes, status, error);
}

fn emit_job_progress_with_total(
    app: &AppHandle,
    job_id: &str,
    downloaded_bytes: i64,
    status: &str,
    error: Option<String>,
) {
    app.emit(
        "download-progress",
        DownloadProgressEvent {
            downloaded_bytes,
            error,
            job_id: job_id.to_string(),
            status: status.to_string(),
            warning: None,
        },
    )
    .ok();
}

fn complete_from_existing_file(app: &AppHandle, job: &DownloadJob) -> Result<bool, String> {
    if target_file_is_complete(&job.target_path, job.size) {
        let warning = checksum_warning_blocking(job)?;
        set_job_completed(&job.id, job.size, warning.as_deref())?;
        let _ = fs::remove_dir_all(parts_dir(&job.target_path));
        emit_job_progress_with_total(app, &job.id, job.size, "completed", None);
        return Ok(true);
    }
    Ok(false)
}

fn scan_existing_completed_files() -> Result<(), String> {
    let jobs = read_download_jobs()?;
    for job in jobs {
        if job.status != "completed" && target_file_is_complete(&job.target_path, job.size) {
            let warning = checksum_warning_blocking(&job)?;
            set_job_completed(&job.id, job.size, warning.as_deref())?;
            let _ = fs::remove_dir_all(parts_dir(&job.target_path));
        }
    }
    Ok(())
}

fn target_file_is_complete(target_path: &str, size: i64) -> bool {
    Path::new(target_path).exists()
        && fs::metadata(target_path)
            .map(|meta| meta.len() as i64)
            .unwrap_or(0)
            == size
}

async fn checksum_warning(job: &DownloadJob) -> Result<Option<String>, String> {
    let Some(expected) = normalized_sha256(job) else {
        return Ok(None);
    };
    let target_path = job.target_path.clone();
    tauri::async_runtime::spawn_blocking(move || checksum_warning_for_path(&target_path, &expected))
        .await
        .map_err(|error| format!("failed to verify sha256: {error}"))?
}

fn checksum_warning_blocking(job: &DownloadJob) -> Result<Option<String>, String> {
    let Some(expected) = normalized_sha256(job) else {
        return Ok(None);
    };
    checksum_warning_for_path(&job.target_path, &expected)
}

fn normalized_sha256(job: &DownloadJob) -> Option<String> {
    job.sha256
        .as_deref()
        .map(str::trim)
        .filter(|value| value.len() == 64)
        .map(|value| value.to_ascii_lowercase())
}

fn checksum_warning_for_path(target_path: &str, expected: &str) -> Result<Option<String>, String> {
    let actual = sha256_file(target_path)?;
    if actual == expected {
        Ok(None)
    } else {
        Ok(Some(format!(
            "SHA256 mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn sha256_file(target_path: &str) -> Result<String, String> {
    let mut file = fs::File::open(target_path)
        .map_err(|error| format!("failed to open file for sha256: {error}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let bytes = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read file for sha256: {error}"))?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn ensure_parts(connection: &rusqlite::Connection, job: &DownloadJob) -> Result<(), String> {
    let part_count = ((job.size + PART_SIZE - 1) / PART_SIZE).max(1);
    for index in 0..part_count {
        let start = index * PART_SIZE;
        let end = (start + PART_SIZE - 1).min(job.size - 1);
        connection
            .execute(
                "insert or ignore into download_parts
                 (job_id, part_index, start_byte, end_byte, downloaded_bytes, status, error)
                 values (?1, ?2, ?3, ?4, 0, 'queued', null)",
                (&job.id, index, start, end),
            )
            .map_err(|error| format!("failed to create download part: {error}"))?;
    }
    Ok(())
}

fn upsert_download_job(connection: &rusqlite::Connection, job: &DownloadJob) -> Result<(), String> {
    connection
        .execute(
            "insert into download_jobs
             (id, repo_id, file_path, file_name, size, downloaded_bytes, status, source_url, target_path, sha256, error, warning, created_at, updated_at)
             values (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, null, null, ?11, ?12)
             on conflict(id) do update set
               source_url = excluded.source_url,
               target_path = excluded.target_path,
               sha256 = coalesce(excluded.sha256, download_jobs.sha256),
               updated_at = excluded.updated_at,
               downloaded_bytes = case when excluded.status = 'completed' then excluded.downloaded_bytes else download_jobs.downloaded_bytes end,
               status = case
                 when excluded.status = 'completed' then 'completed'
                 when download_jobs.status in ('queued', 'downloading', 'paused', 'completed') then download_jobs.status
                 else excluded.status
               end",
            (
                &job.id,
                &job.repo_id,
                &job.file_path,
                &job.file_name,
                job.size,
                job.downloaded_bytes,
                &job.status,
                &job.source_url,
                &job.target_path,
                &job.sha256,
                &job.created_at,
                &job.updated_at,
            ),
        )
        .map_err(|error| format!("failed to save download job: {error}"))?;
    Ok(())
}

fn existing_stable_job_for_target(
    connection: &rusqlite::Connection,
    job: &DownloadJob,
) -> Result<Option<String>, String> {
    let mut statement = connection
        .prepare(
            "select id from download_jobs
             where target_path = ?1
               and size = ?2
               and status in ('queued', 'downloading', 'paused', 'completed')
             limit 1",
        )
        .map_err(|error| format!("failed to prepare duplicate check: {error}"))?;
    match statement.query_row((&job.target_path, job.size), |row| row.get::<_, String>(0)) {
        Ok(id) => Ok(Some(id)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(format!("failed to check duplicate download: {error}")),
    }
}

fn read_download_jobs() -> Result<Vec<DownloadJob>, String> {
    let connection = open_database()?;
    let mut statement = connection
        .prepare(
            "select id, repo_id, file_path, file_name, size, downloaded_bytes, status, source_url, target_path, error, created_at, updated_at, sha256, warning
             from download_jobs
             order by created_at desc",
        )
        .map_err(|error| format!("failed to prepare job list: {error}"))?;
    let jobs = statement
        .query_map([], map_download_job)
        .map_err(|error| format!("failed to read jobs: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to map jobs: {error}"))?;
    Ok(jobs)
}

fn read_active_download_jobs() -> Result<Vec<DownloadJob>, String> {
    let connection = open_database()?;
    let mut statement = connection
        .prepare(
            "select id, repo_id, file_path, file_name, size, downloaded_bytes, status, source_url, target_path, error, created_at, updated_at, sha256, warning
             from download_jobs
             where status in ('queued', 'downloading')
             order by updated_at desc",
        )
        .map_err(|error| format!("failed to prepare active job list: {error}"))?;
    let jobs = statement
        .query_map([], map_download_job)
        .map_err(|error| format!("failed to read active jobs: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to map active jobs: {error}"))?;
    Ok(jobs)
}

fn pause_interrupted_jobs() -> Result<(), String> {
    let connection = open_database()?;
    connection
        .execute(
            "update download_jobs
             set status = 'paused', updated_at = current_timestamp
             where status in ('queued', 'downloading')",
            [],
        )
        .map_err(|error| format!("failed to pause interrupted jobs: {error}"))?;
    connection
        .execute(
            "update download_parts
             set status = 'queued'
             where status = 'downloading'",
            [],
        )
        .map_err(|error| format!("failed to reset interrupted parts: {error}"))?;
    Ok(())
}

fn read_download_job(job_id: &str) -> Result<Option<DownloadJob>, String> {
    let connection = open_database()?;
    let mut statement = connection
        .prepare(
            "select id, repo_id, file_path, file_name, size, downloaded_bytes, status, source_url, target_path, error, created_at, updated_at, sha256, warning
             from download_jobs
             where id = ?1",
        )
        .map_err(|error| format!("failed to prepare job read: {error}"))?;
    match statement.query_row([job_id], map_download_job) {
        Ok(job) => Ok(Some(job)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(format!("failed to read job: {error}")),
    }
}

fn map_download_job(row: &rusqlite::Row<'_>) -> rusqlite::Result<DownloadJob> {
    Ok(DownloadJob {
        id: row.get(0)?,
        repo_id: row.get(1)?,
        file_path: row.get(2)?,
        file_name: row.get(3)?,
        size: row.get(4)?,
        downloaded_bytes: row.get(5)?,
        status: row.get(6)?,
        source_url: row.get(7)?,
        target_path: row.get(8)?,
        error: row.get(9)?,
        created_at: row.get(10)?,
        updated_at: row.get(11)?,
        sha256: row.get(12)?,
        warning: row.get(13)?,
    })
}

fn read_incomplete_parts(job_id: &str) -> Result<Vec<DownloadPart>, String> {
    let connection = open_database()?;
    let mut statement = connection
        .prepare(
            "select part_index, start_byte, end_byte, downloaded_bytes
             from download_parts
             where job_id = ?1 and status != 'completed'
             order by part_index",
        )
        .map_err(|error| format!("failed to prepare parts read: {error}"))?;
    let parts = statement
        .query_map([job_id], |row| {
            Ok(DownloadPart {
                part_index: row.get(0)?,
                start_byte: row.get(1)?,
                end_byte: row.get(2)?,
                downloaded_bytes: row.get(3)?,
            })
        })
        .map_err(|error| format!("failed to read parts: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("failed to map parts: {error}"))?;
    Ok(parts)
}

fn set_job_status(job_id: &str, status: &str, error: Option<&str>) -> Result<(), String> {
    let connection = open_database()?;
    connection
        .execute(
            "update download_jobs set status = ?2, error = ?3, warning = case when ?2 = 'completed' then warning else null end, updated_at = current_timestamp where id = ?1",
            (job_id, status, error),
        )
        .map_err(|error| format!("failed to update job status: {error}"))?;
    Ok(())
}

fn set_job_completed(job_id: &str, size: i64, warning: Option<&str>) -> Result<(), String> {
    let connection = open_database()?;
    connection
        .execute(
            "update download_jobs set status = 'completed', downloaded_bytes = ?2, error = null, warning = ?3, updated_at = current_timestamp where id = ?1",
            (job_id, size, warning),
        )
        .map_err(|error| format!("failed to complete job: {error}"))?;
    Ok(())
}

fn read_job_status(job_id: &str) -> Result<String, String> {
    let connection = open_database()?;
    connection
        .query_row(
            "select status from download_jobs where id = ?1",
            [job_id],
            |row| row.get(0),
        )
        .map_err(|error| format!("failed to read job status: {error}"))
}

fn set_part_progress(
    job_id: &str,
    part_index: i64,
    downloaded_bytes: i64,
    status: &str,
    error: Option<&str>,
) -> Result<(), String> {
    let connection = open_database()?;
    connection
        .execute(
            "update download_parts set downloaded_bytes = ?3, status = ?4, error = ?5 where job_id = ?1 and part_index = ?2",
            (job_id, part_index, downloaded_bytes, status, error),
        )
        .map_err(|error| format!("failed to update part progress: {error}"))?;
    Ok(())
}

fn set_part_completed(job_id: &str, part_index: i64, downloaded_bytes: i64) -> Result<(), String> {
    set_part_progress(job_id, part_index, downloaded_bytes, "completed", None)
}

fn update_job_downloaded(job_id: &str) -> Result<i64, String> {
    let connection = open_database()?;
    let downloaded_bytes: i64 = connection
        .query_row(
            "select coalesce(sum(downloaded_bytes), 0) from download_parts where job_id = ?1",
            [job_id],
            |row| row.get(0),
        )
        .map_err(|error| format!("failed to sum part progress: {error}"))?;
    connection
        .execute(
            "update download_jobs set downloaded_bytes = ?2, updated_at = current_timestamp where id = ?1",
            (job_id, downloaded_bytes),
        )
        .map_err(|error| format!("failed to update job progress: {error}"))?;
    Ok(downloaded_bytes)
}

#[cfg(target_os = "windows")]
fn pick_folder_impl(title: Option<&str>) -> Result<Option<String>, String> {
    let title = title.unwrap_or("Select folder").replace('\'', "''");
    let script = format!(
        r#"
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '{}'
$dialog.ShowNewFolderButton = $true
if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{
  Write-Output $dialog.SelectedPath
}}
"#,
        title
    );

    let output = Command::new("powershell")
        .args(["-NoProfile", "-STA", "-Command", &script])
        .output()
        .map_err(|error| format!("failed to open folder dialog: {error}"))?;

    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let selected = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok((!selected.is_empty()).then_some(selected))
}

#[cfg(not(target_os = "windows"))]
fn pick_folder_impl(_title: Option<&str>) -> Result<Option<String>, String> {
    Err("folder picker is not implemented for this platform yet".to_string())
}

#[cfg(target_os = "windows")]
fn open_path_impl(path: &str) -> Result<(), String> {
    let target = PathBuf::from(path);
    let directory = if target.is_dir() {
        target
    } else {
        target
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "target directory not found".to_string())?
    };
    fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create target directory: {error}"))?;
    Command::new("explorer")
        .arg(directory)
        .spawn()
        .map_err(|error| format!("failed to open directory: {error}"))?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn open_url_impl(url: &str) -> Result<(), String> {
    Command::new("cmd")
        .args(["/C", "start", "", url])
        .spawn()
        .map_err(|error| format!("failed to open url: {error}"))?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn open_path_impl(path: &str) -> Result<(), String> {
    let target = PathBuf::from(path);
    let directory = if target.is_dir() {
        target
    } else {
        target
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "target directory not found".to_string())?
    };
    fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create target directory: {error}"))?;
    Command::new("open")
        .arg(directory)
        .spawn()
        .map_err(|error| format!("failed to open directory: {error}"))?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn open_url_impl(url: &str) -> Result<(), String> {
    Command::new("open")
        .arg(url)
        .spawn()
        .map_err(|error| format!("failed to open url: {error}"))?;
    Ok(())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_path_impl(path: &str) -> Result<(), String> {
    let target = PathBuf::from(path);
    let directory = if target.is_dir() {
        target
    } else {
        target
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| "target directory not found".to_string())?
    };
    fs::create_dir_all(&directory)
        .map_err(|error| format!("failed to create target directory: {error}"))?;
    Command::new("xdg-open")
        .arg(directory)
        .spawn()
        .map_err(|error| format!("failed to open directory: {error}"))?;
    Ok(())
}

#[cfg(all(unix, not(target_os = "macos")))]
fn open_url_impl(url: &str) -> Result<(), String> {
    Command::new("xdg-open")
        .arg(url)
        .spawn()
        .map_err(|error| format!("failed to open url: {error}"))?;
    Ok(())
}

fn build_target_path(models_root: &str, file_path: &str, subfolder: &str) -> PathBuf {
    let mut parts = file_path.split('/').collect::<Vec<_>>();
    let mut target = PathBuf::from(models_root);

    if let Some(first) = parts.first().copied() {
        if KNOWN_MODEL_ROOTS.contains(&first) {
            target.push(first);
            parts.remove(0);
        }
    }

    if !subfolder.trim().is_empty() {
        target.push(sanitize_path_part(subfolder));
    }

    for part in parts {
        target.push(sanitize_path_part(part));
    }

    target
}

fn build_source_url(endpoint: &str, repo_id: &str, file_path: &str) -> String {
    let endpoint = endpoint.trim().trim_end_matches('/');
    let encoded_path = file_path
        .split('/')
        .map(urlencoding::encode)
        .collect::<Vec<_>>()
        .join("/");
    format!("{endpoint}/{repo_id}/resolve/main/{encoded_path}")
}

fn refresh_incomplete_download_sources(endpoint: &str) -> Result<(), String> {
    let connection = open_database()?;
    let jobs = read_download_jobs()?;
    for job in jobs {
        if matches!(job.status.as_str(), "completed" | "canceled") {
            continue;
        }
        let source_url = build_source_url(endpoint, &job.repo_id, &job.file_path);
        connection
            .execute(
                "update download_jobs set source_url = ?2, updated_at = current_timestamp where id = ?1",
                (&job.id, source_url),
            )
            .map_err(|error| format!("failed to refresh download source: {error}"))?;
    }
    Ok(())
}

fn sanitize_path_part(value: &str) -> String {
    value
        .chars()
        .filter(|ch| !matches!(ch, '<' | '>' | ':' | '"' | '|' | '?' | '*'))
        .collect::<String>()
}

fn open_database() -> Result<rusqlite::Connection, String> {
    let base_dir = local_data_dir()?.join("Vanta");
    fs::create_dir_all(&base_dir)
        .map_err(|error| format!("failed to create app data directory: {error}"))?;
    let database_path = base_dir.join("vanta.sqlite");
    let connection = rusqlite::Connection::open(database_path)
        .map_err(|error| format!("failed to open sqlite database: {error}"))?;
    initialize_database(&connection)?;
    Ok(connection)
}

fn initialize_database(connection: &rusqlite::Connection) -> Result<(), String> {
    connection
        .execute(
            "create table if not exists app_kv (
                key text primary key not null,
                value text not null,
                updated_at text not null default current_timestamp
            )",
            [],
        )
        .map_err(|error| format!("failed to initialize app_kv: {error}"))?;
    connection
        .execute(
            "create table if not exists download_jobs (
                id text primary key not null,
                repo_id text not null,
                file_path text not null,
                file_name text not null,
                size integer not null,
                downloaded_bytes integer not null default 0,
                status text not null,
                source_url text not null,
                target_path text not null,
                sha256 text,
                error text,
                warning text,
                created_at text not null,
                updated_at text not null
            )",
            [],
        )
        .map_err(|error| format!("failed to initialize download_jobs: {error}"))?;
    add_column_if_missing(connection, "download_jobs", "sha256", "text")?;
    add_column_if_missing(connection, "download_jobs", "warning", "text")?;
    connection
        .execute(
            "create table if not exists download_parts (
                job_id text not null,
                part_index integer not null,
                start_byte integer not null,
                end_byte integer not null,
                downloaded_bytes integer not null default 0,
                status text not null,
                error text,
                primary key (job_id, part_index)
            )",
            [],
        )
        .map_err(|error| format!("failed to initialize download_parts: {error}"))?;
    Ok(())
}

fn add_column_if_missing(
    connection: &rusqlite::Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<(), String> {
    let escaped_table = table.replace('\'', "''");
    let sql = format!("select count(*) from pragma_table_info('{escaped_table}') where name = ?1");
    let exists: i64 = connection
        .query_row(&sql, [column], |row| row.get(0))
        .map_err(|error| format!("failed to inspect {table}.{column}: {error}"))?;
    if exists == 0 {
        connection
            .execute(
                &format!("alter table {table} add column {column} {definition}"),
                [],
            )
            .map_err(|error| format!("failed to add {table}.{column}: {error}"))?;
    }
    Ok(())
}

fn local_data_dir() -> Result<PathBuf, String> {
    #[cfg(target_os = "windows")]
    {
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .ok_or_else(|| "LOCALAPPDATA is not set".to_string())
    }

    #[cfg(target_os = "macos")]
    {
        std::env::var_os("HOME")
            .map(|home| {
                PathBuf::from(home)
                    .join("Library")
                    .join("Application Support")
            })
            .ok_or_else(|| "HOME is not set".to_string())
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
            Ok(PathBuf::from(path))
        } else {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join(".local").join("share"))
                .ok_or_else(|| "HOME is not set".to_string())
        }
    }
}

fn read_value(connection: &rusqlite::Connection, key: &str) -> Result<Option<String>, String> {
    let mut statement = connection
        .prepare("select value from app_kv where key = ?1")
        .map_err(|error| format!("failed to prepare sqlite read: {error}"))?;
    match statement.query_row([key], |row| row.get::<_, String>(0)) {
        Ok(value) => Ok(Some(value)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(error) => Err(format!("failed to read sqlite value: {error}")),
    }
}

fn write_value(connection: &rusqlite::Connection, key: &str, value: &str) -> Result<(), String> {
    connection
        .execute(
            "insert into app_kv (key, value, updated_at)
             values (?1, ?2, current_timestamp)
             on conflict(key) do update set value = excluded.value, updated_at = current_timestamp",
            (key, value),
        )
        .map_err(|error| format!("failed to write sqlite value: {error}"))?;
    Ok(())
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn hide_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

fn toggle_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        if window.is_visible().unwrap_or(false) {
            let _ = window.hide();
        } else {
            show_main_window(app);
        }
    }
}

fn setup_tray(app: &tauri::App) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, TRAY_SHOW_ID, "显示 Vanta", true, None::<&str>)?;
    let minimize = MenuItem::with_id(app, TRAY_MINIMIZE_ID, "最小化到托盘", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, TRAY_QUIT_ID, "退出", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let menu = Menu::with_items(app, &[&show, &minimize, &separator, &quit])?;
    let icon = app.default_window_icon().cloned();
    let mut builder = TrayIconBuilder::with_id(TRAY_ID)
        .menu(&menu)
        .show_menu_on_left_click(false)
        .tooltip("Vanta\n活动 0\n完成 0\n总数 0\n速度 0 B/s")
        .on_menu_event(|app, event| match event.id().0.as_str() {
            TRAY_SHOW_ID => show_main_window(app),
            TRAY_MINIMIZE_ID => hide_main_window(app),
            TRAY_QUIT_ID => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::DoubleClick {
                button: MouseButton::Left,
                ..
            } = event
            {
                toggle_main_window(tray.app_handle());
            }
        });
    if let Some(icon) = icon {
        builder = builder.icon(icon);
    }
    builder.build(app)?;
    Ok(())
}

#[tauri::command]
fn minimize_main_window(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        window
            .minimize()
            .map_err(|error| format!("failed to minimize main window: {error}"))?;
    }
    Ok(())
}

#[tauri::command]
fn toggle_main_window_maximize(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        if window
            .is_maximized()
            .map_err(|error| format!("failed to read main window maximize state: {error}"))?
        {
            window
                .unmaximize()
                .map_err(|error| format!("failed to unmaximize main window: {error}"))?;
        } else {
            window
                .maximize()
                .map_err(|error| format!("failed to maximize main window: {error}"))?;
        }
    }
    Ok(())
}

#[tauri::command]
fn hide_main_to_tray(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("main") {
        window
            .hide()
            .map_err(|error| format!("failed to hide main window: {error}"))?;
    }
    Ok(())
}

#[tauri::command]
fn quit_app(app: AppHandle) {
    let _ = pause_interrupted_jobs();
    app.exit(0);
}

#[tauri::command]
fn update_tray_summary(app: AppHandle, summary: String) -> Result<(), String> {
    if let Some(tray) = app.tray_by_id(TRAY_ID) {
        tray.set_tooltip(Some(summary))
            .map_err(|error| format!("failed to update tray tooltip: {error}"))?;
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            active_jobs: Mutex::new(HashSet::new()),
        })
        .setup(|app| {
            pause_interrupted_jobs()?;
            setup_tray(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.emit("native-close-requested", ());
                if let Some(webview) = window.app_handle().get_webview_window("main") {
                    let _ = webview.eval(
                        "window.dispatchEvent(new CustomEvent('vanta-native-close-requested'))",
                    );
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            backend_status,
            cancel_download_job,
            enqueue_downloads,
            list_active_download_jobs,
            list_download_jobs,
            load_favorites,
            load_app_storage,
            hide_main_to_tray,
            minimize_main_window,
            pause_download_job,
            pick_folder,
            open_path,
            open_url,
            preview_download_target,
            refresh_download_sources,
            remove_download_job,
            resume_download_job,
            save_access_token,
            save_favorites,
            save_app_settings,
            quit_app,
            toggle_main_window_maximize,
            update_tray_summary
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Vanta");
}
