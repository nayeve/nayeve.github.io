import { useState, useEffect } from "react";
import { AlertTriangle, X } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function SecurityBanner() {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(false);
    }, 10000); // Auto-hide after 10 seconds

    return () => clearTimeout(timer);
  }, []);

  if (!isVisible) return null;

  return (
    <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center">
          <AlertTriangle className="text-yellow-400 h-5 w-5 mr-3 flex-shrink-0" />
          <div className="text-sm">
            <p>
              <strong>Official Authority Information:</strong> This website provides authoritative information on AI applications in amateur radio communications. 
              All technical specifications have been reviewed by communications experts.
            </p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsVisible(false)}
          className="text-yellow-600 hover:text-yellow-800 hover:bg-yellow-100 ml-4"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
