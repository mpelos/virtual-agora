# Rollback Strategy: v1.3 to v1.1

## Overview
This document provides comprehensive rollback procedures for reverting Virtual Agora from v1.3 to v1.1 if critical issues arise during or after migration. Each phase has specific rollback procedures to minimize disruption.

## Rollback Triggers

### Critical Triggers (Immediate Rollback)
1. **Data Loss**: User sessions or reports corrupted
2. **Performance Degradation**: >50% slower than v1.1
3. **Stability Issues**: Crashes or hangs in production
4. **Integration Failures**: Unable to work with existing systems
5. **Blocking Bugs**: Core functionality broken

### Non-Critical Triggers (Planned Rollback)
1. **Minor Bugs**: Non-blocking issues accumulating
2. **User Feedback**: Significant negative response
3. **Resource Constraints**: Unexpected costs or complexity
4. **Timeline Issues**: Migration taking too long

## Pre-Rollback Checklist

Before initiating rollback:

- [ ] Document the specific issue(s) triggering rollback
- [ ] Capture current system state and logs
- [ ] Notify all stakeholders
- [ ] Backup v1.3 data and configuration
- [ ] Identify any irreversible changes
- [ ] Prepare rollback communication

## Phase-Specific Rollback Procedures

### Phase 1: Configuration & State Rollback

#### Configuration Rollback
```python
# scripts/rollback_config.py
import shutil
from datetime import datetime

def rollback_configuration():
    """Revert to v1.1 configuration structure."""
    
    # Backup current config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy("config.yml", f"config_v3_backup_{timestamp}.yml")
    
    # Load v1.3 config
    with open("config.yml", "r") as f:
        v3_config = yaml.safe_load(f)
    
    # Extract v1.1 compatible config
    v1_config = {
        "moderator": v3_config["moderator"],
        "agents": v3_config["agents"]
    }
    
    # Save v1.1 config
    with open("config.yml", "w") as f:
        yaml.dump(v1_config, f)
    
    print("Configuration rolled back to v1.1 structure")
```

#### State Rollback
```python
def rollback_state(checkpoint_file):
    """Revert state to v1.1 compatible structure."""
    
    # Load v1.3 state
    with open(checkpoint_file, "rb") as f:
        v3_state = pickle.load(f)
    
    # Create v1.1 compatible state
    v1_state = {
        # Core fields that exist in both versions
        "session_id": v3_state["session_id"],
        "main_topic": v3_state["main_topic"],
        "current_phase": v3_state["current_phase"],
        "messages": v3_state["messages"],
        "agents": v3_state["agents"],
        "moderator_id": v3_state["moderator_id"],
        
        # Merge specialized agent data back
        "moderator_state": {
            "mode": "facilitation",  # Default mode
            "summaries": v3_state.get("round_summaries", []),
            "reports": v3_state.get("topic_summaries", {})
        }
    }
    
    # Save v1.1 state
    with open(checkpoint_file.replace(".pkl", "_v1.pkl"), "wb") as f:
        pickle.dump(v1_state, f)
```

### Phase 2: Agent Architecture Rollback

#### Code Rollback Steps
1. **Restore ModeratorAgent**
   ```bash
   # Restore from backup
   git checkout v1.1-stable -- src/virtual_agora/agents/moderator.py
   
   # Or from backup directory
   cp backups/moderator_v1.1.py src/virtual_agora/agents/moderator.py
   ```

2. **Remove Specialized Agents**
   ```bash
   # Remove new agent files
   rm src/virtual_agora/agents/summarizer.py
   rm src/virtual_agora/agents/topic_report_agent.py
   rm src/virtual_agora/agents/ecclesia_report_agent.py
   ```

3. **Restore Factory**
   ```bash
   git checkout v1.1-stable -- src/virtual_agora/agents/agent_factory.py
   ```

#### Data Migration
```python
def migrate_specialized_agent_data():
    """Migrate data from specialized agents back to moderator."""
    
    # Collect all summaries
    summaries = []
    if os.path.exists("logs/summarizer.log"):
        summaries = extract_summaries("logs/summarizer.log")
    
    # Collect all reports
    reports = {}
    if os.path.exists("logs/topic_report.log"):
        reports = extract_reports("logs/topic_report.log")
    
    # Inject into moderator state
    return {
        "moderator_summaries": summaries,
        "moderator_reports": reports
    }
```

### Phase 3: Graph Flow Rollback

#### Graph Restoration
```python
# Restore v1.1 graph structure
def restore_v1_graph():
    """Restore v1.1 graph flow."""
    
    # Backup current graph
    shutil.copy(
        "src/virtual_agora/flow/graph.py",
        "src/virtual_agora/flow/graph_v3_backup.py"
    )
    
    # Restore v1.1 files
    files_to_restore = [
        "flow/graph.py",
        "flow/nodes.py", 
        "flow/edges.py"
    ]
    
    for file in files_to_restore:
        src = f"backups/v1.1/{file}"
        dst = f"src/virtual_agora/{file}"
        shutil.copy(src, dst)
```

#### Active Session Handling
```python
def handle_active_sessions():
    """Handle sessions in progress during rollback."""
    
    active_sessions = find_active_sessions()
    
    for session in active_sessions:
        # Option 1: Complete with v1.3
        if session["phase"] >= 4:  # Near completion
            mark_for_completion(session)
            
        # Option 2: Convert to v1.1
        else:
            convert_session_to_v1(session)
            
        # Notify users
        notify_session_owner(session, "rollback_notice.txt")
```

### Phase 4: UI/HITL Rollback

#### UI Component Restoration
```bash
# Restore UI components
git checkout v1.1-stable -- src/virtual_agora/ui/

# Or selective restoration
git checkout v1.1-stable -- src/virtual_agora/ui/human_in_the_loop.py
git checkout v1.1-stable -- src/virtual_agora/ui/components.py
```

#### HITL State Cleanup
```python
def cleanup_hitl_state(state):
    """Remove v1.3 HITL enhancements."""
    
    # Remove new approval types
    if state.get("hitl_state", {}).get("approval_type") in [
        "periodic_stop", 
        "topic_override",
        "agent_poll_override"
    ]:
        state["hitl_state"]["approval_type"] = "continuation"
    
    # Remove periodic stop tracking
    state.pop("periodic_stop_counter", None)
    state.pop("user_stop_history", None)
    
    return state
```

### Phase 5: Test Rollback

#### Test Suite Restoration
```bash
# Restore v1.1 tests
git checkout v1.1-stable -- tests/

# Remove v1.3 specific tests
rm -rf tests/agents/test_*_v3.py
rm -rf tests/migration/
```

## Emergency Rollback Procedure

For immediate rollback in production:

```bash
#!/bin/bash
# emergency_rollback.sh

echo "Starting emergency rollback to v1.1..."

# 1. Stop services
systemctl stop virtual-agora

# 2. Backup current state
timestamp=$(date +%Y%m%d_%H%M%S)
tar -czf "backup_v3_$timestamp.tar.gz" /app/virtual-agora/

# 3. Restore v1.1 codebase
cd /app/virtual-agora
git fetch origin
git checkout v1.1-stable

# 4. Restore v1.1 dependencies
pip install -r requirements_v1.1.txt

# 5. Convert active data
python scripts/emergency_data_conversion.py

# 6. Restart services
systemctl start virtual-agora

echo "Rollback complete. Check logs for any issues."
```

## Data Recovery Procedures

### Session Recovery
```python
def recover_v3_sessions():
    """Recover sessions from v1.3 format."""
    
    v3_sessions = glob.glob("sessions/*_v3.pkl")
    recovered = 0
    failed = []
    
    for session_file in v3_sessions:
        try:
            # Load v1.3 session
            with open(session_file, "rb") as f:
                v3_data = pickle.load(f)
            
            # Convert to v1.1
            v1_data = convert_session_format(v3_data)
            
            # Save as v1.1
            new_file = session_file.replace("_v3.pkl", "_recovered.pkl")
            with open(new_file, "wb") as f:
                pickle.dump(v1_data, f)
            
            recovered += 1
            
        except Exception as e:
            failed.append((session_file, str(e)))
    
    print(f"Recovered: {recovered}, Failed: {len(failed)}")
    return failed
```

### Report Recovery
```python
def recover_reports():
    """Convert v1.3 reports to v1.1 format."""
    
    # v1.3 uses numbered sections
    v3_reports = glob.glob("reports/final_report_*.md")
    
    if v3_reports:
        # Combine into single v1.1 report
        combined_content = []
        
        for report in sorted(v3_reports):
            with open(report, "r") as f:
                combined_content.append(f.read())
        
        # Save as v1.1 format
        with open("reports/final_report.md", "w") as f:
            f.write("\n\n".join(combined_content))
```

## Post-Rollback Verification

### System Health Checks
```python
def verify_rollback():
    """Verify system is functioning on v1.1."""
    
    checks = {
        "config_valid": check_v1_config(),
        "moderator_modes": verify_moderator_modes(),
        "graph_structure": verify_graph_nodes(),
        "ui_components": check_ui_compatibility(),
        "test_suite": run_v1_tests()
    }
    
    failed_checks = [k for k, v in checks.items() if not v]
    
    if failed_checks:
        print(f"Rollback verification failed: {failed_checks}")
        return False
    
    print("Rollback verification successful")
    return True
```

### User Communication
```python
def notify_users_of_rollback():
    """Notify users about rollback."""
    
    template = """
    Subject: Virtual Agora Temporary Rollback to v1.1
    
    Dear User,
    
    We have temporarily rolled back Virtual Agora to version 1.1 due to:
    {reason}
    
    Impact:
    - New features from v1.3 are temporarily unavailable
    - Your sessions have been preserved and converted
    - No data has been lost
    
    Expected resolution: {timeline}
    
    We apologize for any inconvenience.
    """
    
    # Send notifications
    for user in get_active_users():
        send_notification(user, template)
```

## Rollback Prevention

### Monitoring During Migration
```python
def monitor_migration_health():
    """Monitor system health during migration."""
    
    metrics = {
        "response_time": measure_response_time(),
        "error_rate": calculate_error_rate(),
        "memory_usage": get_memory_usage(),
        "user_reports": count_user_issues()
    }
    
    # Alert if thresholds exceeded
    if metrics["error_rate"] > 0.05:  # 5% error rate
        alert("High error rate detected")
    
    if metrics["response_time"] > 2.0:  # 2 second average
        alert("Performance degradation detected")
```

### Gradual Rollout
```python
def gradual_rollout_strategy():
    """Implement gradual rollout to minimize risk."""
    
    stages = [
        {"percentage": 5, "duration": "1 day", "group": "beta_testers"},
        {"percentage": 25, "duration": "3 days", "group": "early_adopters"},
        {"percentage": 50, "duration": "1 week", "group": "half_users"},
        {"percentage": 100, "duration": "ongoing", "group": "all_users"}
    ]
    
    for stage in stages:
        enable_for_percentage(stage["percentage"])
        monitor_for_duration(stage["duration"])
        
        if rollback_triggered():
            return rollback_to_previous()
```

## Lessons Learned Documentation

After any rollback, document:

1. **Trigger Event**
   - What specifically caused the rollback?
   - When was it detected?
   - Who made the decision?

2. **Impact Assessment**
   - How many users affected?
   - What data needed conversion?
   - What functionality was lost?

3. **Recovery Process**
   - How long did rollback take?
   - Were there any issues?
   - What could be improved?

4. **Prevention Measures**
   - What testing missed this issue?
   - How can we prevent similar issues?
   - What monitoring is needed?

## Recovery Timeline

| Phase | Action | Duration | Recovery Point |
|-------|--------|----------|----------------|
| 1 | Stop Services | 5 min | No data loss |
| 2 | Code Rollback | 15 min | Git checkpoint |
| 3 | Data Conversion | 30 min | Last backup |
| 4 | Verification | 20 min | Health checks |
| 5 | Restart | 10 min | Service online |
| **Total** | **Full Rollback** | **80 min** | **Operational** |

## Support During Rollback

- **Hotline**: Emergency support number
- **Slack Channel**: #virtual-agora-rollback
- **Documentation**: This guide + runbooks
- **Escalation**: Team leads → CTO if needed

---

**Document Version**: 1.0
**Last Updated**: [Current Date]
**Status**: Emergency Procedure